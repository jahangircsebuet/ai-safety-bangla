import csv

INPUT_CSV = "/home/malam10/projects/ai-safety-bangla/rebuttal/hatespeech_v1_to_prompt.csv"
OUT_NOT_USEFUL_SAMPLES = "not_useful_as_question_samples.csv"
OUT_USEFUL_SAMPLES = "useful_as_question_samples.csv"
OUT_NOT_VALID_SAMPLES = "not_a_valid_question_samples.csv"

# 🔹 NEW OUTPUT FILES
OUT_SAFE_PROMPTS = "hs_to_converted_safe_prompts.csv"
OUT_UNSAFE_PROMPTS = "hs_to_converted_unsafe_prompts.csv"



with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    not_useful_rows = []
    not_valid_question_rows = []
    useful_rows = []
    

    # 🔹 NEW BUCKETS
    safe_prompt_rows = []
    unsafe_prompt_rows = []

    for row in reader:
        # 🔴 REMOVE malformed extra columns
        if None in row:
            del row[None]

        # Existing logic (UNCHANGED)
        if row.get("prompt") == "Not useful as question":
            not_useful_rows.append(row)
        elif row.get("prompt") == "Not a valid question":
            not_valid_question_rows.append(row)
        else:
            useful_rows.append(row)

        # 🔹 NEW LOGIC: filter by prompt_safety
        prompt_safety = row.get("prompt_safety", "").strip().lower()
        not_useful = row.get("prompt") == "Not useful as question"
        not_valid_question = row.get("prompt") == "Not a valid question"

        if not not_useful and not not_valid_question and prompt_safety == "safe":
            safe_prompt_rows.append(row)
        elif not not_useful and not not_valid_question and prompt_safety == "unsafe":
            unsafe_prompt_rows.append(row)

# =======================
# EXISTING OUTPUT FILES
# =======================

with open(OUT_NOT_USEFUL_SAMPLES, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(not_useful_rows)

with open(OUT_NOT_VALID_SAMPLES, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(not_valid_question_rows)

with open(OUT_USEFUL_SAMPLES, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(useful_rows)

# =======================
# 🔹 NEW OUTPUT FILES
# =======================

with open(OUT_SAFE_PROMPTS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(safe_prompt_rows)

with open(OUT_UNSAFE_PROMPTS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(unsafe_prompt_rows)

print(f"Saved {len(not_useful_rows)} rows → {OUT_NOT_USEFUL_SAMPLES}")
print(f"Saved {len(useful_rows)} rows → {OUT_USEFUL_SAMPLES}")
print(f"Saved {len(safe_prompt_rows)} rows → {OUT_SAFE_PROMPTS}")
print(f"Saved {len(unsafe_prompt_rows)} rows → {OUT_UNSAFE_PROMPTS}")
