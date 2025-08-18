import json

# Step 1: Read the input JSON file
input_file_path = "catqa_bangla.json"  # Replace with your actual file path

with open(input_file_path, "r", encoding="utf-8") as f:
    nested_data = json.load(f)

# Step 2: Convert to flat format
flattened_prompts = []
id_counter = 1

for parent_cat, sub_cats in nested_data.items():
    for sub_cat, prompts in sub_cats.items():
        for prompt in prompts:
            flattened_prompts.append({
                "id": str(id_counter),
                "prompt": prompt,
                "category": parent_cat,
                "sub_category": sub_cat
            })
            id_counter += 1

# Step 3: Output structure
output = {"prompts": flattened_prompts}

# Step 4: Save to output file (optional)
output_file_path = "converted_catqa_bn.json"
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Print a preview
print(json.dumps(output, indent=2, ensure_ascii=False))