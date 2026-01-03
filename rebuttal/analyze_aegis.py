import json
import pandas as pd

# =========================
# Load AEGIS JSON file
# =========================
json_path = "/home/malam10/projects/ai-safety-bangla/rebuttal/aegis_v2_extracted_with_severity.json"   # or aegis.jsonl

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# =========================
# Use prompt_bn as sentence
# =========================
df["sentence"] = df["prompt_bn"].astype(str)

# =========================
# Sentence length (word count)
# =========================
df["sentence_length"] = df["sentence"].apply(lambda x: len(x.split()))

# =========================
# Length bins
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
# Length distribution
# =========================
length_distribution = df["length_range"].value_counts().sort_index()

print("Sentence Length Distribution:")
print(length_distribution)

# =========================
# Safety / Hate counts
# =========================
safety_counts = df["prompt_label"].value_counts()

print("\nPrompt Safety Counts:")
print(safety_counts)

# =========================
# Length × Safety Crosstab
# =========================
length_vs_safety = pd.crosstab(
    df["length_range"],
    df["prompt_label"],
    rownames=["Sentence Length Range"],
    colnames=["Prompt Label"]
)

print("\nSentence Length vs Prompt Safety:")
print(length_vs_safety)
