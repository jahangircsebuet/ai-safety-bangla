import json
from copy import deepcopy
import re

INPUT_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/hatespeech_to_prompt_final.json"
OUTPUT_FILE = "/home/malam10/projects/ai-safety-bangla/final/data/hatespeech_dataset_formatted.json"

# hatespeech dataset fields 
#   {
#     "text",
#     "label",
#     "explanation",
#     "prompt", #rename into prompt_bn
#     "harm_score",
#     "prompt_safety",
#     "prompt_safety_reason",
#     "response",
#     "aegis_category",
#     "aegis_reason",
#     "aegis_category_name"
#   },


FIELDS_TO_MOVE = [
    "harm_score",
    "prompt_safety",
    "prompt_safety_reason",
    "response", #rename into response_bn
    "aegis_category",
    "aegis_reason",
    "aegis_category_name"
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

    # change prompt fieldname into prompt_bn
        # change prompt fieldname into prompt_bn
    if "prompt" in record:
        record["prompt_bn"] = record.pop("prompt")

    return record


def main():
    # Load input JSON
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")

    # # Transform records
    # transformed = [transform_record(obj) for obj in data]

    
    # discard prompts with length below 5
    # discard records which are "Not useful as question"
    # Transform records
    transformed = [transform_record(obj) for obj in data if obj.get("prompt") != "Not useful as question" and len(re.findall(r"\S+", obj.get("prompt").strip())) >= 5]

    # Write output JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(transformed)} records")
    print(f"📁 Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

