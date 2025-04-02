import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import spacy
from geopy.geocoders import Nominatim
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# ---------------------------
# Load Environment Variables
# ---------------------------
# Specify the path to your .env file
dotenv_path = '/Users/chenzhiq/.mytoken_env'
load_dotenv(dotenv_path=dotenv_path)

# load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------------------
# Initialize Tools
# ---------------------------
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="mandalay_earthquake_locator")

# ---------------------------
# Helper Functions
# ---------------------------
def search_news(query, from_date, to_date, page=1):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "pageSize": 100,
        "page": page,
        "apiKey": NEWS_API_KEY,
        "language": "en"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("NewsAPI error:", response.text)
        return None

def extract_clean_text(article):
    """
    Combines and cleans the 'title', 'description', and 'content' fields
    using BeautifulSoup to remove HTML tags.
    """
    parts = []
    for field in ['title', 'description', 'content']:
        text = article.get(field, "")
        if text:
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
            parts.append(clean_text)
    return " ".join(parts)

def extract_clean_text_with_timeout(article, timeout=5):
    """
    Runs extract_clean_text with a timeout. If the extraction takes longer than
    'timeout' seconds, returns an empty string.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_clean_text, article)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            print("Text extraction timed out. Skipping full text for this article.")
            return ""

def extract_locations_spacy(text):
    doc = nlp(text)
    locations = set()
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            locations.add(ent.text)
    return list(locations)

def geocode_location(location_name):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except Exception as e:
        print("Geocoding error:", e)
        return (None, None)

def download_image(url, save_folder, filename, timeout=5):
    """
    Downloads an image from a URL with a timeout.
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            file_path = os.path.join(save_folder, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return file_path
        else:
            print(f"⚠️ Failed to download image {url} — Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"❗ Image download failed: {e} — Skipping {url}")
        return None


def main():
    texts_folder = "texts"
    images_folder = "images"
    os.makedirs(texts_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    query_keywords = "Mandalay earthquake OR Myanmar earthquake"
    date_list = ["2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31", "2025-04-01"]

    all_records = []
    global_idx = 1

    for date_str in date_list:
        from_date = date_str
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        to_date_obj = date_obj + timedelta(days=1)
        to_date = to_date_obj.strftime("%Y-%m-%d")
        print(f"\n🔍 Searching articles for date: {from_date}")
        
        day_article_count = 0
        page = 1
        while True:
            news_data = search_news(query_keywords, from_date, to_date, page)
            if news_data and news_data.get("articles"):
                articles = news_data.get("articles")
                if not articles:
                    break
                for article in articles:
                    start_time = time.time()

                    # Extract and clean full text (with timeout)
                    cleaned_text = extract_clean_text_with_timeout(article, timeout=5)
                    if not cleaned_text.strip():
                        print(f"❗ Skipped article {global_idx} on {from_date} — text extraction failed or timed out.")
                        global_idx += 1
                        continue

                    # Save text
                    text_filename = f"text_{global_idx}.txt"
                    text_filepath = os.path.join(texts_folder, text_filename)
                    with open(text_filepath, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)

                    # Extract location info
                    locations = extract_locations_spacy(cleaned_text)
                    lat, lon = geocode_location(locations[0]) if locations else (None, None)

                    # Handle image URLs
                    if article.get("images") and isinstance(article.get("images"), list) and article.get("images"):
                        image_urls = article.get("images")
                    elif article.get("urlToImage"):
                        image_urls = [article.get("urlToImage")]
                    else:
                        image_urls = [None]

                    image_files = []
                    for img_idx, image_url in enumerate(image_urls, start=1):
                        if image_url:
                            image_filename = f"image_{global_idx}_{img_idx}.jpg"
                            local_image_path = download_image(image_url, images_folder, image_filename)
                            if local_image_path:
                                image_files.append(local_image_path)
                        else:
                            image_files.append("")

                    # Record entry
                    record = {
                        "idx": global_idx,
                        "url": article.get("url"),
                        "title": article.get("title"),
                        "date": from_date,
                        "text_file": text_filename,
                        "image_files": ", ".join(image_files),
                        "extracted_locations": ", ".join(locations) if locations else "",
                        "latitude": lat,
                        "longitude": lon
                    }

                    all_records.append(record)
                    day_article_count += 1
                    elapsed = time.time() - start_time
                    print(f"✅ Processed article {global_idx} in {elapsed:.2f}s for date {from_date}")
                    global_idx += 1
                    time.sleep(1)

                if len(articles) < 100:
                    break
                page += 1
                time.sleep(1)
            else:
                break

        print(f"📅 Finished {from_date}: {day_article_count} articles successfully processed.")

    # Append to existing CSV if it exists
    df = pd.DataFrame(all_records)
    record_file = "entry_record.csv"
    if os.path.exists(record_file):
        existing_df = pd.read_csv(record_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(record_file, index=False)
    print(f"\n📁 Entry record saved to {record_file}")


if __name__ == "__main__":
    main()
