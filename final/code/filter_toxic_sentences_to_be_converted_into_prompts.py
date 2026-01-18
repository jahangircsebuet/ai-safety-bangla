import pandas as pd
import json
import re

import unicodedata

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    return text



def store_common():
    import pandas as pd
    import re
    import unicodedata

    # ---------------- PATHS ----------------
    CSV_1 = "/home/malam10/projects/ai-safety-bangla/final/data/converted_toxic_texts_only.csv"
    CSV_2 = "/home/malam10/projects/ai-safety-bangla/final/data/original_csv_texts_only.csv"

    COMMON_OUT = "/home/malam10/projects/ai-safety-bangla/rebuttal/common.csv"
    UNCOMMON_OUT = "/home/malam10/projects/ai-safety-bangla/rebuttal/uncommon.csv"

    TEXT_COL = "text"
    # --------------------------------------

    def normalize_text(text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    # Load CSVs
    df1 = pd.read_csv(CSV_1)
    df2 = pd.read_csv(CSV_2)

    # Normalize text
    df1[TEXT_COL] = df1[TEXT_COL].astype(str).apply(normalize_text)
    df2[TEXT_COL] = df2[TEXT_COL].astype(str).apply(normalize_text)

    # Create sets
    set1 = set(df1[TEXT_COL])
    set2 = set(df2[TEXT_COL])

    # Common and uncommon
    common_texts = sorted(set1 & set2)
    uncommon_texts = sorted(set1 - set2)

    # Save results
    pd.DataFrame({"text": common_texts}).to_csv(COMMON_OUT, index=False, encoding="utf-8")
    pd.DataFrame({"text": uncommon_texts}).to_csv(UNCOMMON_OUT, index=False, encoding="utf-8")

    # Summary
    print("==== CSV Comparison Summary ====")
    print("CSV 1 texts      :", len(set1))
    print("CSV 2 texts      :", len(set2))
    print("Common texts     :", len(common_texts))
    print("Uncommon texts   :", len(uncommon_texts))
    print("Saved:", COMMON_OUT)
    print("Saved:", UNCOMMON_OUT)

def get_text_from_toxic_csv():
    import pandas as pd

    CSV_PATH = "/home/malam10/projects/ai-safety-bangla/final/data/toxic_comments_gte_10_lte_40_tokens.csv"
    OUT_CSV = "/home/malam10/projects/ai-safety-bangla/final/data/original_csv_texts_only.csv"

    # Load CSV
    df_csv = pd.read_csv(CSV_PATH)

    # Extract only the text column
    df_text_only = df_csv[["text"]]

    # Save to CSV
    df_text_only.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("Saved texts:", len(df_text_only))
    print("Output file:", OUT_CSV)

def get_covnerted_text_only():
    JSON_PATH = "/home/malam10/projects/ai-safety-bangla/final/data/toxic_to_prompt_deduplicated.json"
    OUT_CSV = "/home/malam10/projects/ai-safety-bangla/final/data/converted_toxic_texts_only.csv"

    # Load JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        converted_items = json.load(f)

    # Extract text field safely
    texts = [
        item["text"]
        for item in converted_items
        if isinstance(item, dict) and "text" in item and isinstance(item["text"], str)
    ]

    # Save to CSV
    df = pd.DataFrame({"text": texts})
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("Saved converted texts:", len(df))
    print("Output file:", OUT_CSV)

# ---------------- PATHS ----------------
CSV_PATH = "/home/malam10/projects/ai-safety-bangla/final/data/toxic_comments_gte_10_lte_40_tokens.csv"
JSON_PATH = "/home/malam10/projects/ai-safety-bangla/final/data/toxic_to_prompt_deduplicated.json"
OUTPUT_CSV = "/home/malam10/projects/ai-safety-bangla/final/unconverted_rows_for_prompt_generation.csv"

TEXT_COL = "text"
# -------------------------------------

# Load CSV
df_csv = pd.read_csv(CSV_PATH)

# Load JSON (list of dicts)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    converted_items = json.load(f)


get_covnerted_text_only()
get_text_from_toxic_csv()
store_common()
#     # Normalize CSV
# df_csv[TEXT_COL] = df_csv[TEXT_COL].apply(normalize_text)

# # Normalize JSON
# converted_texts = {
#     normalize_text(item[TEXT_COL])
#     for item in converted_items
#     if TEXT_COL in item
# }

converted_text_list = [
    item["text"]
    for item in converted_items
    if isinstance(item, dict) and "text" in item
]

print("Total extracted texts:", len(converted_text_list))
print("First 5 samples:")
for t in converted_text_list[:5]:
    print(repr(t))


# # Extract converted texts into a set for fast lookup
converted_texts = {
    item[TEXT_COL].strip()
    for item in converted_items
    if TEXT_COL in item and isinstance(item[TEXT_COL], str)
}
print("CSV rows:", len(df_csv))
print("JSON items:", len(converted_items))

print("Unique CSV texts:", df_csv[TEXT_COL].nunique())
print("Unique JSON texts:", len(converted_texts))

# # Normalize CSV text
df_csv[TEXT_COL] = df_csv[TEXT_COL].astype(str).str.strip()

# Filter rows NOT converted
df_unconverted = df_csv[~df_csv[TEXT_COL].isin(converted_texts)]

# Save to CSV
df_unconverted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

# Summary
print("==== Conversion Audit Summary ====")
print(f"Total CSV rows        : {len(df_csv)}")
print(f"Converted prompts     : {len(converted_texts)}")
print(f"Unconverted rows saved: {len(df_unconverted)}")
print(f"Output file           : {OUTPUT_CSV}")
