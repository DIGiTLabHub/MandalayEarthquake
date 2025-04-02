# Enhanced gallery data generator with tagging, sentiment analysis, and discrepancy score

import os
import json
import base64
import pandas as pd
import time
import ast
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
dotenv_path = '/Users/chenzhiq/.mytoken_env'
load_dotenv(dotenv_path=dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
ENTRY_CSV = "entry_record.csv"
TEXTS_DIR = "texts"
IMAGES_DIR = "images"
OUTPUT_JSON = "gallery_data.json"

# Load entry records
df = pd.read_csv(ENTRY_CSV)
entries = []

def compute_discrepancy(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return round(1 - sim, 3)  # Discrepancy: 1 - similarity

# For statistics
summary_tags = {"Damaged Building": 0, "Injury/Death": 0, "Recovery": 0}
image_tags = {"Damaged Building": 0, "Injury/Death": 0, "Recovery": 0}

for idx, row in df.iterrows():
    start_time = time.time()
    print(f"Processing entry {idx + 1}/{len(df)}...")

    image_files = row.get("image_files", "")
    if not isinstance(image_files, str) or not image_files.strip():
        continue

    text_path = os.path.join(TEXTS_DIR, row["text_file"])
    image_path = image_files.split(",")[0].strip()
    if not os.path.exists(text_path) or not os.path.exists(image_path):
        print(f"Skipping missing file: {text_path} or {image_path}")
        continue

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    title = row.get("title", "")
    url = row.get("url", "")
    date = row.get("date", "")
    lat = row.get("latitude", None)
    lon = row.get("longitude", None)

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

        # --- GPT: Sentiment Analysis ---
        sentiment_prompt = (
            "Given the following summary, classify its sentiment as one of: Neutral, Concerned, Hopeful, Distressing, Tragic.\n"
            f"Summary: {summary_text}"
        )

        sentiment_response = client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a sentiment analysis assistant.",
            input=sentiment_prompt
        )
        sentiment = sentiment_response.output_text.strip()

    except Exception as e:
        print(f"Error generating summary/sentiment for {image_path}: {e}")
        summary_text = "News Summary: [Failed to generate]"
        sentiment = "Unknown"

    # --- GPT: Generate Caption & Tags ---
    try:
        caption_prompt = (
            "Describe this image in one caption-style sentence focused on earthquake effects such as damage, "
            "collapse, rescue, or people affected.\n"
            "Add a short tag in parentheses at the beginning.\n\n"
            "Also, list relevant tags (e.g., Damaged Building, People, Rescue, Debris, Injured People) "
            "in the format: Relevant Tags: [tag1, tag2, ...]"
        )

        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{img_base64}"

            caption_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant generating disaster image captions and tags."},
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
        tag_line = next((line for line in caption_text.splitlines() if line.startswith("Tags:")), "")
        tags = []
        if tag_line:
            raw_tags = tag_line.replace("Tags:", "").strip()
            try:
                tags = ast.literal_eval(raw_tags)
                if not isinstance(tags, list):
                    tags = []
            except Exception:
                print(f"⚠️ Warning: Couldn't parse tags for {image_path}. Got: {raw_tags}")

    except Exception as e:
        print(f"Error generating caption/tags for {image_path}: {e}")
        caption_text = "AI Image Caption: [Failed to generate]"
        tags = []

    # --- Discrepancy Score ---
    discrepancy_score = compute_discrepancy(summary_text, caption_text)

    # --- Stats Tracking ---
    for tag in tags:
        if "damage" in tag.lower(): image_tags["Damaged Building"] += 1
        if "injur" in tag.lower() or "death" in tag.lower(): image_tags["Injury/Death"] += 1
        if "rescue" in tag.lower() or "recovery" in tag.lower(): image_tags["Recovery"] += 1
    if "damage" in summary_text.lower(): summary_tags["Damaged Building"] += 1
    if "injur" in summary_text.lower() or "death" in summary_text.lower(): summary_tags["Injury/Death"] += 1
    if "rescue" in summary_text.lower() or "recovery" in summary_text.lower(): summary_tags["Recovery"] += 1

    entries.append({
        "image_file": image_path,
        "title": title,
        "url": url,
        "date": date,
        "latitude": lat,
        "longitude": lon,
        "summary": summary_text,
        "sentiment": sentiment,
        "caption": caption_text,
        "tags": tags,
        "discrepancy_score": discrepancy_score
    })

    duration = round(time.time() - start_time, 2)
    print(f"Finished entry {idx + 1} in {duration} seconds\n")

# Save JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

# Save statistics
stats = {
    "summary_tags": summary_tags,
    "image_tags": image_tags
}
with open("gallery_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"✅ Finished generating {len(entries)} gallery entries with tagging, sentiment, and discrepancy.")
