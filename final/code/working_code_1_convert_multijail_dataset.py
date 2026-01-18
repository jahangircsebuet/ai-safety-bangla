
import csv
import json
from datasets import load_dataset

# # ***************bn column extraction start

# ds = load_dataset("DAMO-NLP-SG/MultiJail")

# print(ds['train'].column_names)

# bn_data = ds['train']['bn']
# print(bn_data[:5])  # Print first 5 Bengali prompts

# # Define the output CSV file path
# output_csv = "multijail_bn_updated.csv"

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

# # ***************bn column extraction end



# ========================================
# create multijail_en_bn.csv starts
# ========================================
# maping from list of tags to single most severe tag starts here

# ==========================
# 1. Severity Ranking
# ==========================
# 🎯 Why this ordering works =====> get references of the below points

# This ranking follows:

# UNICEF & global content safety law (child & sexual abuse highest)

# Anthropic Constitutional AI severity order

# OpenAI safety spec v2

# Google DeepMind safety taxonomy

# ICLR’24 MultiJail red-teaming risk level alignment

# Legal/ethical severity thresholds

# severity_order = [
#     "Child abuse",
#     "Sexual exploitation & human trafficking",
#     "Self-harm",
#     "Terrorism & organized crime",
#     "Violence & incitement",
#     "Weapons",
#     "Animal abuse",
#     "Property crime & vandalism",
#     "Theft",
#     "Fraud & deception",
#     "Soliciting personally identifiable information (PII)",
#     "Hate speech & offensive language",
#     "Discrimination & injustice",
#     "Bullying & harassment",
#     "Adult content",
#     "Substance abuse & banned substances",
#     "Conspiracy theories & misinformation",
#     "Non-violent unethical behavior"
# ]

# # ==========================
# # 2. Mapping tag → severity rank
# # ==========================

# severity_map = {tag: rank for rank, tag in enumerate(severity_order, start=1)}

# # ==========================
# # 3. Function to pick highest severity tag
# # ==========================

# def select_single_tag(tag_list):
#     """
#     Given a list of tags from MultiJail, choose the single most severe tag.
#     Severity: lower rank = more severe.
#     """
#     if not tag_list:
#         return None

#     valid_tags = [t for t in tag_list if t in severity_map]
#     if not valid_tags:
#         return None

#     # Return tag with minimum rank (highest severity)
#     return min(valid_tags, key=lambda t: severity_map[t])


# # maping from list of tags to single most severe tag ends here

# # code to read english and bangla text and tags, source info from the dataset starts 

# from datasets import load_dataset
# import csv
# import ast

# # Load dataset
# ds = load_dataset("DAMO-NLP-SG/MultiJail")

# # Check available columns
# print(ds['train'].column_names)

# # Output CSV
# output_csv = "multijail_en_bn.csv"

# with open(output_csv, mode="w", newline='', encoding="utf-8") as file:
#     writer = csv.writer(file)
    
#     # Write header
#     writer.writerow(["text_en", "text_bn", "source", "tags"])
    
#     # Write rows
#     for example in ds['train']:
#         bn_text = example['bn']        # Bengali text
#         en_text = example['en']        # English text
#         source = example['source']     # Source info
#         tags = example['tags']         # Original tags list
        
#         # Convert string representation → list
#         tag_str = example["tags"]  # e.g. "['Child abuse', 'Sexual exploitation & human trafficking']"
#         tag_list = ast.literal_eval(tag_str)

#         # Pick most severe tag
#         single_tag = select_single_tag(tag_list)

#         writer.writerow([en_text, bn_text, source, tags, single_tag])

# print(f"Data successfully written to {output_csv}")

# ========================================
# create multijail_en_bn.csv ends 
# ========================================


# # *******************store multijail into foarmatted json start 
# # Input and output file paths

# csv_path = "datasets/multijail_bn.csv"         # correct relative path
# json_path = "datasets/converted_multijail_bn.json"  # save output alongside it

# # Start ID
# start_id = 1

# # Output list
# converted_data = []

# # Read CSV and convert each line
# with open(csv_path, mode='r', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     next(reader)  # skip header if present

#     for i, row in enumerate(reader):
#         prompt_text = row[0].strip()  # assuming prompt is in first column
#         # If you want to derive category from a column, use row[1] or row[2] etc.
#         entry = {
#             "id": str(start_id + i),
#             "prompt": prompt_text,
#             "category": "জেইলব্রেক"  # or dynamically: row[1] if present
#         }
#         converted_data.append(entry)

# # Save to JSON file
# with open(json_path, "w", encoding="utf-8") as outfile:
#     json.dump(converted_data, outfile, ensure_ascii=False, indent=2)

# print(f"✅ Converted {len(converted_data)} entries and saved to {json_path}")


# # *******************store multijail into foarmatted json end



# ================================================================================
# # ******************* store multijail (EN + BN) into formatted JSON start
# ================================================================================
import csv
import json

# Input and output file paths
csv_path = "/home/malam10/projects/ai-safety-bangla/final/data/multijail_en_bn.csv"  
json_path = "/home/malam10/projects/ai-safety-bangla/final/data/converted_multijail_en_bn.json"

# Start ID
start_id = 1

# Output list
converted_data = []

# Read CSV and convert each line
with open(csv_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    for i, row in enumerate(reader):
        en_text = row[0].strip()
        bn_text = row[1].strip()

        entry = {
            "id": str(start_id + i),
            "prompt_en": en_text,
            "prompt_bn": bn_text,
            "source": row[2].strip(),  # assuming source is in third column
            "tags": row[3].strip(),    # assuming tags are in fourth column
            "tag": row[4].strip(),     # most severe tag in fifth column
            "category": "jailbreak"   # or your Bengali label: "জেইলব্রেক"
        }

        converted_data.append(entry)

# Save to JSON
with open(json_path, "w", encoding="utf-8") as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=2)

print(f"✅ Converted {len(converted_data)} entries and saved to {json_path}")
# ================================================================================
# # ******************* store multijail (EN + BN) into formatted JSON end
# ================================================================================
