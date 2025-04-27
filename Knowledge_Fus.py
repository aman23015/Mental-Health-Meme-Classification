import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


def generate_knowledge_fusion_embeddings(
    input_json_path,
    output_embedding_path,
    output_index_path,
    model_name="all-mpnet-base-v2"
):
    """
    Generates knowledge fusion embeddings for memes using OCR, reasoning, and visual text.

    Args:
        input_json_path (str): Path to the JSON file with keys:
            - "ocr_text"
            - "visual_description"
            - "figurative_reasoning" with subkeys:
                - "cause_effect"
                - "figurative_understanding"
                - "mental_state"
        output_embedding_path (str): Path to save the numpy embedding matrix (.npy)
        output_index_path (str): Path to save the index-to-filename mapping (.json)
        model_name (str): SentenceTransformer model name (default: all-mpnet-base-v2)
    """

    # === Load your annotated JSON file ===
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # === Load Sentence-Transformer (Π) ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # === Prepare list of embeddings ===
    embeddings = []
    filenames = []

    for img_name, info in tqdm(data.items(), desc="Embedding memes"):
        ocr_text = info.get("ocr_text", "")
        visual_description = info.get("visual_description", "")
        reasoning = info.get("figurative_reasoning", {})

        # Concatenate all parts of reasoning
        full_reasoning = " ".join([
            reasoning.get("cause_effect", ""),
            reasoning.get("figurative_understanding", ""),
            reasoning.get("mental_state", "")
        ])

        # Get individual embeddings using GPU
        e_o = model.encode(ocr_text, convert_to_numpy=True, device=device)
        e_r = model.encode(full_reasoning, convert_to_numpy=True, device=device)
        e_v = model.encode(visual_description, convert_to_numpy=True, device=device)

        # Concatenate into one vector E_k = [e_o | e_r | e_v]
        e_k = np.concatenate([e_o, e_r, e_v])
        embeddings.append(e_k)
        filenames.append(img_name)

    # === Convert to final embedding matrix ===
    E = np.stack(embeddings)  # Shape: (n, 3d) = (num_memes, 2304)
    print("Final Knowledge Fusion DB shape:", E.shape)

    # === Save embeddings and corresponding filenames ===
    np.save(output_embedding_path, E)

    with open("embedding_index.json", "w") as f:
        json.dump(filenames, f)

    print("Saved knowledge fusion DB to 'knowledge_fusion_embeddings.npy'")

generate_knowledge_fusion_embeddings(
    input_json_path="/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/anxiety_train_with_reasoning.json",
    output_embedding_path="knowledge_fusion_embeddings.npy",
    output_index_path="embedding_index.json"
)
