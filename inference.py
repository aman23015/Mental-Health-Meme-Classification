import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import numpy as np
import json
import os
from main import RAGAnxietyMemeDataset, collate_fn  # Adjust this import according to your training file

# === Inference Function ===
def inference(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predictions

# === Main Function for Inference ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (Update these according to your setup)
    json_path = "/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/anxiety_test_with_reasoning.json"
    csv_path = "/home/aaditya23006/AMAN/NLP/DATA/anxiety_test.csv"
    embedding_path = "/home/aaditya23006/AMAN/NLP/DATA/Knowledge_Fusion/knowledge_fusion_embeddings.npy"
    index_path = "/home/aaditya23006/AMAN/NLP/DATA/Knowledge_Fusion/embedding_index.json"
    best_model_path = "/home/aaditya23006/AMAN/NLP/outputs/best_model.pt"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=7  # ensure num_labels matches training
    ).to(device)

    # Load best trained model weights
    model.load_state_dict(torch.load(best_model_path))
    print("✅ Best trained model loaded successfully.")

    # Create test dataset and dataloader
    test_dataset = RAGAnxietyMemeDataset(
        json_path=json_path,
        csv_path=csv_path,
        embedding_path=embedding_path,
        index_path=index_path,
        k=3
    )

    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # Run inference
    true_labels, predictions = inference(model, test_loader, device)

    # Calculate Metrics
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    weighted_f1 = f1_score(true_labels, predictions, average="weighted")

    print("\n📊 Classification Report:\n")
    print(classification_report(true_labels, predictions, target_names=list(test_dataset.label_to_idx.keys())))
    
    print(f"🚀 Macro F1 Score: {macro_f1:.4f}")
    print(f"🚀 Weighted F1 Score: {weighted_f1:.4f}")

    results = {
        "true_labels": [int(label) for label in true_labels],
        "predictions": [int(pred) for pred in predictions],
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }

    os.makedirs("/home/aaditya23006/AMAN/NLP/outputs", exist_ok=True)
    with open("/home/aaditya23006/AMAN/NLP/outputs/inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Inference results saved to 'inference_results.json'.")

if __name__ == "__main__":
    main()
