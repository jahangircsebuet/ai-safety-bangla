
import csv
import json
from datasets import load_dataset

# ***************bn column extraction start

# ds = load_dataset("DAMO-NLP-SG/MultiJail")

# print(ds['train'].column_names)

# bn_data = ds['train']['bn']
# print(bn_data[:5])  # Print first 5 Bengali prompts

# # Define the output CSV file path
# output_csv = "multijail_bn.csv"

# # Open the CSV file for writing
# with open(output_csv, mode="w", newline='', encoding="utf-8") as file:
#     writer = csv.writer(file)
    
#     # Write header
#     writer.writerow(["text"])
    
#     # Write each row
#     for example in ds['train']:
#         bn_text = example['bn']

#         writer.writerow([bn_text])

# print(f"Data successfully written to {output_csv}")

# ***************bn column extraction end




# *******************store multijail into foarmatted json start 
# Input and output file paths
csv_path = "multijail_bn.csv"   # replace with your actual CSV file
json_path = "converted_multijail_bn.json"

# Start ID
start_id = 1

# Output list
converted_data = []

# Read CSV and convert each line
with open(csv_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # skip header if present

    for i, row in enumerate(reader):
        prompt_text = row[0].strip()  # assuming prompt is in first column
        # If you want to derive category from a column, use row[1] or row[2] etc.
        entry = {
            "id": str(start_id + i),
            "prompt": prompt_text,
            "category": "জেইলব্রেক"  # or dynamically: row[1] if present
        }
        converted_data.append(entry)

# Save to JSON file
with open(json_path, "w", encoding="utf-8") as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=2)

print(f"✅ Converted {len(converted_data)} entries and saved to {json_path}")


# *******************store multijail into foarmatted json end

