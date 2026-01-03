import pandas as pd

# =========================
# Load CSV
# =========================
csv_path = "/home/malam10/projects/ai-safety-bangla/rebuttal/bd_shs.csv"   # change if needed
df = pd.read_csv(csv_path)

# Ensure correct column names
df.columns = ["sentence", "target", "type", "hate_speech"]

# =========================
# Sentence Length (word count)
# =========================
df["sentence_length"] = df["sentence"].astype(str).apply(lambda x: len(x.split()))

# =========================
# Define length bins
# =========================
bins = [0, 5, 10, 15, 20, 25, float("inf")]
labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "25+"]

df["length_range"] = pd.cut(
    df["sentence_length"],
    bins=bins,
    labels=labels,
    right=True
)

# =========================
# Count sentences per length range
# =========================
length_distribution = df["length_range"].value_counts().sort_index()

print("Sentence Length Distribution:")
print(length_distribution)

# =========================
# Hate speech counts
# =========================
hate_counts = df["hate_speech"].value_counts().rename({
    0: "Not Hate Speech",
    1: "Hate Speech"
})

print("\nHate Speech Counts:")
print(hate_counts)


length_vs_hate = pd.crosstab(
    df["length_range"],
    df["hate_speech"],
    rownames=["Sentence Length Range"],
    colnames=["Hate Speech Label"]
)

print("\nSentence Length vs Hate Speech:")
print(length_vs_hate)


import pandas as pd

# =========================
# Load BD SHS CSV
# =========================
csv_path = "bd_shs.csv"   # change if needed
df = pd.read_csv(csv_path)

# =========================
# Sentence length (word count)
# =========================
df["sentence_length"] = df["sentence"].astype(str).apply(lambda x: len(x.split()))

# =========================
# Filter: length >= 10 AND hate speech only
# =========================
df_hate_10_plus = df[
    (df["sentence_length"] >= 10) &
    (df["hate speech"] == 1)
]

# =========================
# Summary
# =========================
print(f"Total samples: {len(df)}")
print(f"Hate speech samples (length >= 10): {len(df_hate_10_plus)}")

# =========================
# Save filtered dataset
# =========================
df_hate_10_plus.to_csv("bd_shs_hate_len10_plus.csv", index=False)
print("✅ Filtered dataset saved to bd_shs_hate_len10_plus.csv")
