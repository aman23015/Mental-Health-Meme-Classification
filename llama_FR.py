import json
import time
from tqdm import tqdm
from openai import OpenAI

def generate_figurative_reasoning(
    input_json_path,
    output_json_path,
    api_key,
    base_url="https://api.studio.nebius.com/v1/",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    max_tokens=512,
    temperature=0.2,
    top_p=0.7,
    sleep_time=1.0
):
    """
    Generate figurative reasoning triples (cause-effect, figurative understanding, mental state)
    using a LLaMA model served on NVIDIA NIM.

    Args:
        input_json_path (str): Path to input JSON with OCR and visual descriptions.
        output_json_path (str): Path to save output JSON with reasoning.
        api_key (str): NVIDIA NIM API key.
        base_url (str): NIM base URL (default NVIDIA integration).
        model_name (str): Model ID to use from NIM.
        max_tokens (int): Max tokens to generate per response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        sleep_time (float): Delay between requests to avoid rate limits.
    """

    # === Setup OpenAI-compatible NIM client ===
    client = OpenAI(
        base_url= base_url,
        api_key= api_key
    )

    # === Load input file ===
    with open(input_json_path, "r") as f:
        data = json.load(f)

    output_data = {}

    for img_name, info in tqdm(data.items(), desc="Generating reasoning with LLaMA"):
        ocr_text = info.get("ocr_text", "")
        visual_description = info.get("visual_description", "")

        prompt = f"""
    Analyze the following anxiety meme image to extract commonsense reasoning in the form of triples. These relationships should capture the following elements:

    1. Cause-Effect: Identify concrete causes or results of the situation depicted in the meme.
    2. Figurative Understanding: Capture underlying metaphors, analogies, or symbolic meanings that convey the meme’s deeper message, including any ironic or humorous undertones.
    3. Mental State: Capture specific mental or emotional states depicted in the meme.

    Visual Description: {visual_description}
    Meme Text: {ocr_text}

    Respond in the format:
    Cause-Effect: ...
    Figurative Understanding: ...
    Mental State: ...
    """.strip()

        try:
            # Call the LLaMA model
            response = client.chat.completions.create(
                model= model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature= temperature,
                top_p= top_p,
                max_tokens= max_tokens
            )

            reply = response.choices[0].message.content

            # Basic extraction (could be improved with regex if needed)
            sections = {
                "cause_effect": "",
                "figurative_understanding": "",
                "mental_state": ""
            }

            for line in reply.split("\n"):
                line_lower = line.lower().strip()
                if "cause-effect" in line_lower:
                    sections["cause_effect"] = line.split(":", 1)[-1].strip()
                elif "figurative understanding" in line_lower:
                    sections["figurative_understanding"] = line.split(":", 1)[-1].strip()
                elif "mental state" in line_lower:
                    sections["mental_state"] = line.split(":", 1)[-1].strip()

            output_data[img_name] = {
                "visual_description": visual_description,
                "ocr_text": ocr_text,
                "figurative_reasoning": sections
            }

            time.sleep(sleep_time)  # To avoid rate limits
        except Exception as e:
            output_data[img_name] = {
                "visual_description": visual_description,
                "ocr_text": ocr_text,
                "figurative_reasoning": {
                    "cause_effect": f"ERROR: {str(e)}",
                    "figurative_understanding": "",
                    "mental_state": ""
                }
            }

    # === Save output ===
    with open("output_json_path", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved figurative reasoning to {output_json_path}")


generate_figurative_reasoning(
    input_json_path="/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/anxiety_test_vd_td.json",
    output_json_path="/home/aaditya23006/AMAN/NLP/DATA/JSON_FILES/anxiety_test_with_reasoning.json",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMjM1MTgwNDA3NjMxMDUzODU2NiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwMjYwMTI1MSwidXVpZCI6ImM4NzQyMjc5LWRiODktNGIwOS05ZmMxLTEyNDZhMDViZDU4YSIsIm5hbWUiOiJVbm5hbWVkIGtleSIsImV4cGlyZXNfYXQiOiIyMDMwLTA0LTE2VDIwOjIwOjUxKzAwMDAifQ.5vRjXXbvOX61-jnlV14BImDkns3GDU4btSl6SWR1UFU"
)
