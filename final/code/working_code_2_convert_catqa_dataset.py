import json

# # Step 1: Read the input JSON file
# input_file_path = "datasets/catqa_bangla.json"  # Replace with your actual file path

# with open(input_file_path, "r", encoding="utf-8") as f:
#     nested_data = json.load(f)

# # Step 2: Convert to flat format
# flattened_prompts = []
# id_counter = 1

# for parent_cat, sub_cats in nested_data.items():
#     for sub_cat, prompts in sub_cats.items():
#         for prompt in prompts:
#             flattened_prompts.append({
#                 "id": str(id_counter),
#                 "prompt": prompt,
#                 "category": parent_cat,
#                 "sub_category": sub_cat
#             })
#             id_counter += 1

# # Step 3: Output structure
# output = {"prompts": flattened_prompts}

# # Step 4: Save to output file (optional)
# output_file_path = "converted_catqa_bn.json"
# with open(output_file_path, "w", encoding="utf-8") as f:
#     json.dump(output, f, indent=2, ensure_ascii=False)

# # Print a preview
# print(json.dumps(output, indent=2, ensure_ascii=False))


# code to merge english and bangla catqa datasets starts 

import json
from datasets import load_dataset

# ===========================
# Step 1: Load your Bangla prompts
# ===========================
bn_file = "/home/malam10/projects/ai-safety-bangla/final/data/translated_catqa.json"

with open(bn_file, "r", encoding="utf-8") as f:
    bn_data = json.load(f)["prompts"]   # list of dicts


# ===========================
# Step 2: Load English CatQA dataset (HF)
# ===========================
en_ds = load_dataset("declare-lab/CategoricalHarmfulQA")["en"]


# ds = load_dataset("declare-lab/CategoricalHarmfulQA")
# print(ds)

# Sanity check
assert len(bn_data) == len(en_ds), "Bangla and English dataset sizes do NOT match!"


# ===========================
# Step 3: Merge English + Bangla
# ===========================
merged = []

for i, bn_item in enumerate(bn_data):

    en_item = en_ds[i]   # English item index = Bangla item index

    merged.append({
        "id": bn_item["id"],
        "prompt_bn": bn_item["prompt"],
        "prompt_en": en_item["Question"],
        "category": bn_item["category"],       # same category as BN
        "sub_category": bn_item["sub_category"]
    })


# ===========================
# Step 4: Save merged JSON
# ===========================
output_file = "/home/malam10/projects/ai-safety-bangla/final/data/converted_catqa_en_bn.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully merged! Saved to {output_file}")

# code to merge english and bangla catqa datasets ends
