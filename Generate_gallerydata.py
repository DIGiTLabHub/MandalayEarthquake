import os
import json
import base64
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
ENTRY_CSV = "entry_record.csv"
TEXTS_DIR = "texts"
IMAGES_DIR = "images"
OUTPUT_JSON = "gallery_data.json"

# Load entry records
df = pd.read_csv(ENTRY_CSV)
entries = []

for idx, row in df.iterrows():
    image_files = row.get("image_files", "")
    if not isinstance(image_files, str) or not image_files.strip():
        continue  # Skip if no image

    text_path = os.path.join(TEXTS_DIR, row["text_file"])
    image_path = image_files.split(",")[0].strip()  # Use the first image

    if not os.path.exists(text_path) or not os.path.exists(image_path):
        print(f"Skipping missing file: {text_path} or {image_path}")
        continue

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    title = row.get("title", "")
    url = row.get("url", "")
    date = row.get("date", "")

    # --- GPT: Generate News Summary ---
    try:
        summary_prompt = (
            "Using the title and article text below, write a 1–2 sentence summary focused on "
            "building damage, collapse, rescue efforts, or people impacted by the 2025 Mandalay earthquake. "
            "Label the result as 'News Summary:'.\n\n"
            f"Title: {title}\n\nText: {text}"
        )

        summary_response = client.responses.create(
            model="gpt-4o-mini",
            instructions="You are an assistant generating disaster news summaries.",
            input=summary_prompt
        )
        summary_text = summary_response.output_text.strip()

    except Exception as e:
        print(f"Error generating summary for {image_path}: {e}")
        summary_text = "News Summary: [Failed to generate]"

    # --- GPT: Generate Image Caption ---
    try:
        caption_prompt = (
            "Describe this image in one caption-style sentence focused on earthquake effects such as damage, "
            "collapse, rescue, or people affected.\n"
            "Add a short tag in parentheses at the beginning to indicate confidence and type, "
            "e.g., (Likely - Rescue Scene), (Certain - Building Collapse), (Uncertain - People Impacted)."
        )

        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{img_base64}"

            caption_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant generating disaster image captions."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]
            )

        caption_text = caption_response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        caption_text = "AI Image Caption: [Failed to generate]"

    entries.append({
        "image_file": image_path,
        "title": title,
        "url": url,
        "date": date,
        "summary": summary_text,
        "caption": caption_text
    })

# Save JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"✅ Finished generating {len(entries)} gallery entries to {OUTPUT_JSON}")
