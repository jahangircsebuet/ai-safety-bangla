#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter and Translate Aegis AI Safety Dataset to Bangla
------------------------------------------------------
✓ Loads NVIDIA Aegis-AI-Content-Safety-Dataset-2.0
✓ Filters out samples with unsafe responses (robust to missing fields)
✓ Translates prompt, response, and category → Bangla (facebook/nllb-200-distilled-600M)
✓ Writes results incrementally after each batch
✓ Always starts fresh — no resume logic
"""

import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===========================================================
# CONFIG
# ===========================================================
DATASET_NAME = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
OUTPUT_PATH = "/home/tahad/ai-safety-bangla/datasets/Ageis_unsafe_responses_translated.json"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================
# LOAD DATASET
# ===========================================================
print(f" Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")

# ---- Filter entries with response_label == "unsafe" safely ----
filtered = []
for ex in dataset:
    label = ex.get("response_label")
    if isinstance(label, str) and label.lower() == "unsafe":
        filtered.append(ex)

print(f" Filtered entries with unsafe responses: {len(filtered)}")

# ===========================================================
# LOAD TRANSLATION MODEL
# ===========================================================
print(f"Loading translation model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# ===========================================================
# TRANSLATION FUNCTION
# ===========================================================
def translate_batch(texts, tgt_lang="ben_Beng"):
    """Translate a batch of English texts to Bangla."""
    if not texts:
        return []
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)

    bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_new_tokens=512,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ===========================================================
# TRANSLATION LOOP (ALWAYS STARTS FRESH)
# ===========================================================
output = []
print(f" Starting translation of {len(filtered)} unsafe responses...")

for i in tqdm(range(0, len(filtered), BATCH_SIZE), desc="Translating"):
    batch = filtered[i:i + BATCH_SIZE]

    prompts = [b.get("prompt", "") or "" for b in batch]
    responses = [b.get("response", "") or "" for b in batch]
    categories = [b.get("violated_categories", "") or "" for b in batch]

    try:
        prompt_bn = translate_batch(prompts)
        response_bn = translate_batch(responses)
        category_bn = translate_batch(categories)
    except Exception as e:
        print(f"Error in batch {i}: {e}")
        continue

    for j, ex in enumerate(batch):
        translated_entry = {
            "prompt": prompt_bn[j],
            "response": response_bn[j],
            "prompt_label": ex.get("prompt_label", ""),
            "response_label": ex.get("response_label", ""),
            "category_label": ex.get("violated_categories", ""),
        }
        output.append(translated_entry)

    # Write results incrementally after each batch
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f" Saved {len(output)} entries so far to {OUTPUT_PATH}")

print("\n Translation complete!")
print(f"Output saved to: {OUTPUT_PATH}")
print(f"Total translated entries: {len(output)}")
