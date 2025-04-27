import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def generate_visual_descriptions(image_dir, ocr_json_path, output_path):
    """
    Generate visual captions using BLIP for each image based on the meme image.
    
    Args:
        image_dir (str): Path to the directory containing images.
        ocr_json_path (str): Path to JSON file with OCR text {filename: ocr_text}.
        output_path (str): Path to save the output JSON with visual + OCR data.
    """
    # === Load OCR JSON ===
    with open(ocr_json_path, "r") as f:
        ocr_data = json.load(f)

    # === Load BLIP model ===
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Process each image ===
    anxiety_train_vd_td = {}
    for filename, ocr_text in tqdm(ocr_data.items(), desc="Generating visual descriptions"):
        image_path = os.path.join(image_dir, filename)
        print(image_path)
        # input("wait")
        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            visual_caption = processor.decode(output[0], skip_special_tokens=True)

            anxiety_train_vd_td[filename] = {
                "visual_description": visual_caption,
                "ocr_text": ocr_text
            }
        except Exception as e:
            anxiety_train_vd_td[filename] = {
                "visual_description": f"ERROR: {str(e)}",
                "ocr_text": ocr_text
            }

    # === Save result ===
    with open(output_path, "w") as f:
        json.dump(anxiety_train_vd_td, f, indent=2)

    print(f"Saved output to: {output_path}")


generate_visual_descriptions(
    image_dir = "/home/aaditya23006/AMAN/NLP/DATA/anxiety_train_image",
    ocr_json_path = "/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/ocr_anxiety_train.json",
    output_path = "anxiety_train_vd_td.json"
)


generate_visual_descriptions(
    image_dir = "/home/aaditya23006/AMAN/NLP/DATA/anxiety_test_image",
    ocr_json_path = "/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/ocr_anxiety_test.json",
    output_path = "anxiety_train_vd_td.json"
)