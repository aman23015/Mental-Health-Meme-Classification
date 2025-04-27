import os
import easyocr
import json
from tqdm import tqdm
from PIL import Image

def extract_ocr_text(image_folder, output_json_name):
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=True)  # set gpu=True if available

    # Dictionary to store results
    ocr_text_by_image = {}

    # Iterate through images in the folder
    for filename in tqdm(os.listdir(image_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_folder, filename)
            
            # Run OCR
            result = reader.readtext(image_path, detail=0)  # detail=0 returns just text
            
            # Join text fragments
            joined_text = " ".join(result)
            
            # Store in dictionary
            ocr_text_by_image[filename] = joined_text

    # Save the OCR result to the specified JSON file
    output_json_path = f'{output_json_name}.json'
    with open(output_json_path, 'w') as f:
        json.dump(ocr_text_by_image, f, indent=2)

    print(f"OCR extraction complete. Results saved to {output_json_path}")

# Example usage:
test_image_folder_path = '/home/aaditya23006/AMAN/NLP/DATA/anxiety_test_image'  # Path to your images
train_image_folder_path = '/home/aaditya23006/AMAN/NLP/DATA/anxiety_train_image'  # Path to your images
test_output_json_name = 'ocr_anxiety_test'  # Output JSON filename without extension
train_output_json_name = 'ocr_anxiety_train'  # Output JSON filename without extension

extract_ocr_text(train_image_folder_path, train_output_json_name)
extract_ocr_text(test_image_folder_path, test_output_json_name)