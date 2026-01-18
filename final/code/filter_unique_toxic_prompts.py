import json

INPUT_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/toxic_to_prompt_deduplicated.json"
UNIQUE_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/filtered_final_set_of_converted_toxic_to_prompt.json"
DUPLICATE_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/duplicate_items_(toxic_to_prompt)_log.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

seen_texts = set()
unique_items = []
duplicate_items = []

for item in data:
    text = item.get("text", "").strip()

    if not text:
        # Optional: treat empty text as duplicate
        duplicate_items.append(item)
        continue

    if text in seen_texts:
        duplicate_items.append(item)
    else:
        seen_texts.add(text)
        unique_items.append(item)

with open(UNIQUE_FILE, "w", encoding="utf-8") as f:
    json.dump(unique_items, f, ensure_ascii=False, indent=2)

with open(DUPLICATE_FILE, "w", encoding="utf-8") as f:
    json.dump(duplicate_items, f, ensure_ascii=False, indent=2)

print("=== Deduplication Summary ===")
print(f"Total items     : {len(data)}")
print(f"Unique items    : {len(unique_items)}")
print(f"Duplicate items : {len(duplicate_items)}")
print(f"Saved unique → {UNIQUE_FILE}")
print(f"Saved duplicates → {DUPLICATE_FILE}")
