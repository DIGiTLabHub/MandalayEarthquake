import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from custom path
dotenv_path = '/Users/chenzhiq/.mytoken_env'
load_dotenv(dotenv_path=dotenv_path)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def classify_entry(summary, caption):
    prompt = f"""
Given the following information from an earthquake-related image, classify the degree of LOSS and RESILIENCE.

--- INFORMATION ---
News Summary: "{summary}"
Image Caption: "{caption}"

--- LOSS CLASSIFICATION RULES ---
Loss Level 3 (High): Mentions of injured/dead people or fully collapsed buildings with visible debris.
Loss Level 2 (Moderate): Partially collapsed buildings and other descriptions of damage.
Loss Level 1 (None): No visible or described damage.

--- RESILIENCE CLASSIFICATION RULES ---
Resilience Level 3 (High): Many rescue or medical personnel are involved.
Resilience Level 2 (Moderate): One or a few people doing search or rescue.
Resilience Level 1 (None): No visible recovery effort or rescue work.

Return your answer in the format:
Loss Level: [1-3]
Resilience Level: [1-3]
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # Use regex to extract the levels
    loss_match = re.search(r"Loss Level:\s*(\d)", content)
    resilience_match = re.search(r"Resilience Level:\s*(\d)", content)

    if loss_match and resilience_match:
        return int(loss_match.group(1)), int(resilience_match.group(1))
    else:
        print(f"[WARNING] Could not parse response:\n{content}\n")
        return None, None

# Load your gallery data
with open("gallery_data.json", "r") as f:
    data = json.load(f)

# Process and classify
for i, item in enumerate(data):
    summary = item.get("summary", "")
    caption = item.get("caption", "")
    print(f"[record #{i+1}] {item.get('image_file', '')} ...")

    try:
        loss, resilience = classify_entry(summary, caption)
    except Exception as e:
        print(f"Error: {e}")
        loss, resilience = None, None
    item["lossLevel"] = loss
    item["resilienceLevel"] = resilience

# Save result
with open("gallery_data_augmented.json", "w") as f:
    json.dump(data, f, indent=2)

print("Classification complete. Saved to gallery_data_augmented.json")
