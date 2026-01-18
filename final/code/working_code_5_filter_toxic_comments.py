import pandas as pd
import re

# ---------- CONFIG ----------
INPUT_CSV = "/home/malam10/projects/ai-safety-bangla/rebuttal/toxic_comments.csv"
OUTPUT_CSV = "/home/malam10/projects/ai-safety-bangla/rebuttal/toxic_comments_gte_10_lte_40_tokens.csv"
MIN_TOKENS = 10
MAX_TOKENS = 40
# ----------------------------


def bangla_tokenize(text: str):
    """
    Tokenize Bangla text into lexical tokens.
    Keeps only Bangla Unicode characters.
    """
    if not isinstance(text, str):
        return []
    # Bangla Unicode block: \u0980–\u09FF
    if text == "সাফা কবিরের চেহারায় কেউ জুতা মার...":
        print("Found specific text! and its tokens: ", re.findall(r'[\u0980-\u09FF]+', text))
        print("length: ", len(re.findall(r'[\u0980-\u09FF]+', text)))
    tokens = re.findall(r'[\u0980-\u09FF]+', text)
    return tokens

# Read CSV
df = pd.read_csv(INPUT_CSV)

# Count Bangla lexical tokens
df["bangla_token_count"] = df["text"].apply(
    lambda x: len(bangla_tokenize(x))
)

# Filter sentences with LESS than 10 Bangla tokens
filtered_df = df[df["bangla_token_count"] >= MIN_TOKENS].copy()
filtered_df = filtered_df[filtered_df["bangla_token_count"] <= MAX_TOKENS].copy()

# Drop helper column if not needed
filtered_df.drop(columns=["bangla_token_count"], inplace=True)

# Save result
filtered_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Saved {len(filtered_df)} rows with >= {MIN_TOKENS} and <= {MAX_TOKENS} Bangla tokens to {OUTPUT_CSV}")
