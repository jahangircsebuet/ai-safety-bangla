import json
from copy import deepcopy

INPUT_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/multijail_dataset_augmented.json"
OUTPUT_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/multijail_dataset_augmented_formatted.json"

FIELDS_TO_MOVE = [
    "response_bn",
    "aegis_category",
    "aegis_reason",
    "aegis_category_name",
    "prompt_safety",
    "harm_score"
]

MODEL_NAME = "chatgpt"


def transform_record(record):
    record = deepcopy(record)

    # Build closed_llm entry
    llm_entry = {}

    for field in FIELDS_TO_MOVE:
        if field in record:
            llm_entry[field] = record.pop(field)

    # Add model name
    llm_entry["model_name"] = MODEL_NAME

    # Attach closed_llm list
    record["closed_llm"] = [llm_entry]

    return record


def main():
    # Load input JSON
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")

    # Transform records
    transformed = [transform_record(obj) for obj in data]

    # Write output JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(transformed)} records")
    print(f"📁 Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

