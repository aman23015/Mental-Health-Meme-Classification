import os
import json
import torch
import pandas as pd
import numpy as np
from torch import nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RAGAnxietyMemeDataset(Dataset):
    def __init__(self, json_path, csv_path, embedding_path, index_path, k=3):
        self.k = k

        # === Load annotated meme reasoning text
        with open(json_path, "r") as f:
            self.text_data = json.load(f)

        # === Load label CSV
        df = pd.read_csv(csv_path)
        df["sample_id"] = df["sample_id"].apply(lambda x: x if x.endswith(".jpg") else f"{x}.jpg")

        # === Load Knowledge Fusion Embeddings
        self.embedding_matrix = np.load(embedding_path)

        # === Load the index mapping (i.e., row i → filename)
        with open(index_path, "r") as f:
            self.embedding_index = json.load(f)

        # === Sentence encoder for query text embedding
        self.encoder = SentenceTransformer("all-mpnet-base-v2")

        # === Label mapping
        unique_labels = sorted(df["meme_anxiety_categories"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # === Build data entries aligned with labels
        self.entries = []
        for _, row in df.iterrows():
            filename = row["sample_id"]
            label = row["meme_anxiety_categories"]
            if filename in self.text_data:
                self.entries.append({
                    "filename": filename,
                    "label": self.label_to_idx[label]
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        filename = entry["filename"]
        label = entry["label"]
        item = self.text_data[filename]
        reasoning = item.get("figurative_reasoning", {})

        query_text = "\n".join([
            f"OCR Text: {item.get('ocr_text', '')}",
            f"Visual Description: {item.get('visual_description', '')}",
            f"Cause-Effect: {reasoning.get('cause_effect', '')}",
            f"Figurative Understanding: {reasoning.get('figurative_understanding', '')}",
            f"Mental State: {reasoning.get('mental_state', '')}"
        ])
        
        # Get query embedding
        full_reasoning = " ".join([
            reasoning.get("cause_effect", ""),
            reasoning.get("figurative_understanding", ""),
            reasoning.get("mental_state", "")
        ])
        e_o = self.encoder.encode(item.get("ocr_text", ""), convert_to_numpy=True)
        e_r = self.encoder.encode(full_reasoning, convert_to_numpy=True)
        e_v = self.encoder.encode(item.get("visual_description", ""), convert_to_numpy=True)
        query_embedding = np.concatenate([e_o, e_r, e_v])

        # Compute cosine similarities
        sims = cosine_similarity([query_embedding], self.embedding_matrix)[0]

        # Get top-k (excluding itself)
        top_k_indices = sims.argsort()[-(self.k + 1):][::-1]  # descending order
        top_k = [i for i in top_k_indices if self.embedding_index[i] != filename][:self.k]
        # print("top_k ",top_k)
        # input("wait")

        # Get retrieved texts
        retrieved_texts = []
        for i in top_k:
            retrieved_filename = self.embedding_index[i]
            retrieved_item = self.text_data.get(retrieved_filename, {})
            r_reasoning = retrieved_item.get("figurative_reasoning", {})
        
            retrieved_ocr = retrieved_item.get("ocr_text", "")
            retrieved_vis = retrieved_item.get("visual_description", "")
            retrieved_cause = r_reasoning.get("cause_effect", "")
            retrieved_fig = r_reasoning.get("figurative_understanding", "")
            retrieved_mental = r_reasoning.get("mental_state", "")
        
            retrieved_text = "\n".join([
                # f"[Retrieved Example]",
                f"OCR Text: {retrieved_ocr}",
                f"Visual Description: {retrieved_vis}",
                f"Cause-Effect: {retrieved_cause}",
                f"Figurative Understanding: {retrieved_fig}",
                f"Mental State: {retrieved_mental}"
            ])
        
            retrieved_texts.append(retrieved_text)


        return {
            "text_input": query_text,
            "retrieved": retrieved_texts,  # list of top-k examples
            "label": label
        }


# train_dataset = RAGAnxietyMemeDataset(
#     json_path="/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/anxiety_train_with_reasoning.json",
#     csv_path="/home/aaditya23006/AMAN/NLP/DATA/anxiety_train.csv",
#     embedding_path="/home/aaditya23006/AMAN/NLP/DATA/Knowledge_Fusion/knowledge_fusion_embeddings.npy", #Knowledge fusion path
#     index_path="/home/aaditya23006/AMAN/NLP/DATA/Knowledge_Fusion/embedding_index.json",
#     k=3  # Number of top-k retrieved examples
# )

# sample = train_dataset[0]
# print(sample["text_input"])            # Original meme content
# print(sample["retrieved"])         # First retrieved similar example
# print("Label (int):", sample["label"])  # Integer label

def collate_fn(batch, tokenizer):
    input_texts = []
    labels = []

    for item in batch:
        full_prompt = item["text_input"] + "\n\n" + "\n\n".join(item["retrieved"])
        input_texts.append(full_prompt)
        labels.append(item["label"])

    tokenized = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": torch.tensor(labels)
    }


# === Training Function ===
def train(model, tokenizer, train_loader, val_loader, epochs, optimizer, output_dir, device):
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    train_losses, val_losses = [], []

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training loop
        model.train()
        total_train_loss = 0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_macro_f1 = f1_score(train_labels, train_preds, average="macro")
        train_weighted_f1 = f1_score(train_labels, train_preds, average="weighted")

        train_losses.append(avg_train_loss)

        print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Macro F1: {train_macro_f1:.4f}, Weighted F1: {train_weighted_f1:.4f}")

        # Validation loop
        model.eval()
        val_preds, val_labels = [], []
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro")
        val_weighted_f1 = f1_score(val_labels, val_preds, average="weighted")

        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Macro F1: {val_macro_f1:.4f}, Weighted F1: {val_weighted_f1:.4f}")

        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print("✅ Best model saved.")

    # Plot losses
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.show()
    print(f"✅ Loss curve saved to {output_dir}/loss_curve.png")

# === Main Function ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (adjust as necessary)
    json_path = "/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/anxiety_train_with_reasoning.json"
    csv_path = "/home/aaditya23006/AMAN/NLP/DATA/anxiety_train.csv"
    embedding_path = "/home/aaditya23006/AMAN/NLP/DATA/Knowledge_Fusion/knowledge_fusion_embeddings.npy"
    index_path = "/home/aaditya23006/AMAN/NLP/DATA/Knowledge_Fusion/embedding_index.json"
    output_dir = "/home/aaditya23006/AMAN/NLP/outputs"

    # Load dataset
    dataset = RAGAnxietyMemeDataset(
        json_path=json_path,
        csv_path=csv_path,
        embedding_path=embedding_path,
        index_path=index_path,
        k=3
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(dataset.label_to_idx)
    ).to(device)

    # Data loaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 10  # adjust as needed

    # Start training
    train(model, tokenizer, train_loader, val_loader, epochs, optimizer, output_dir, device)

if __name__ == "__main__":
    main()
