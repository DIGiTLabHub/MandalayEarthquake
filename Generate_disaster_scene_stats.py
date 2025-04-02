import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# --------------------------------------
# Load the JSON data
# --------------------------------------
with open("gallery_data_augmented.json", "r") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# --------------------------------------
# Clean and Extract Sentiment Labels
# --------------------------------------
sentiment_raw = df["sentiment"].dropna()

# Extract label inside **...**, clean punctuation, standardize casing
sentiment_cleaned = (
    sentiment_raw
    .str.extract(r"\*\*(.*?)\*\*")[0]
    .str.strip()
    .str.strip(string.punctuation)
    .str.capitalize()
)

# Custom sentiment order: most severe to most positive
sentiment_order = ["Tragic", "Distressing", "Concerned", "Hopeful"]

# --------------------------------------
# Load and Clean Loss and Resilience Levels
# --------------------------------------
loss_levels = df["lossLevel"].dropna().astype(int)
resilience_levels = df["resilienceLevel"].dropna().astype(int)

# Set Seaborn style
sns.set(style="whitegrid")

# --------------------------------------
# Plot 1: Sentiment Histogram (Custom Order)
# --------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(
    x=sentiment_cleaned,
    hue=sentiment_cleaned,
    order=sentiment_order,
    palette="Blues",
    legend=False
)
plt.title("Histogram of Sentiment Levels")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------
# Plot 2: Loss Level Histogram
# --------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=loss_levels, hue=loss_levels, palette="Reds", legend=False)
plt.title("Histogram of Loss Levels")
plt.xlabel("Loss Level (1=None, 2=Moderate, 3=High)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------------------------
# Plot 3: Resilience Level Histogram
# --------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=resilience_levels, hue=resilience_levels, palette="Greens", legend=False)
plt.title("Histogram of Resilience Levels")
plt.xlabel("Resilience Level (1=None, 2=Moderate, 3=High)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --------------------------------------
# Print Loss vs Resilience Matrix
# --------------------------------------
matrix = pd.crosstab(loss_levels, resilience_levels)
print("\nJoint Matrix: Loss Level vs Resilience Level")
print(matrix)
