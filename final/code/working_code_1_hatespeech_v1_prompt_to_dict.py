import csv
import json

INPUT_CSV = "/home/malam10/projects/ai-safety-bangla/rebuttal/hatespeech_v1_to_prompt.csv"  # change path if needed
OUTPUT_JSON = "/home/malam10/projects/ai-safety-bangla/rebuttal/hatespeech_v1_prompt_to_dict_all.json"   # output JSON file

rows_as_dicts = []

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # 🔴 Remove malformed extra columns if present
        if None in row:
            del row[None]

        rows_as_dicts.append(row)

# Save to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
    json.dump(rows_as_dicts, jf, ensure_ascii=False, indent=2)

print(f"Saved {len(rows_as_dicts)} records → {OUTPUT_JSON}")

