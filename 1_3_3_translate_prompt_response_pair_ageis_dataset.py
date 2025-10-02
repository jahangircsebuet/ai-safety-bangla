#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translate and export filtered Aegis prompt/response pairs to Bangla, keeping category labels.

Run (example):
CUDA_VISIBLE_DEVICES=3 nohup python 1_3_3_translate_prompt_response_pair_ageis_dataset.py \
  > /home/tahad/ai-safety-bangla/logs/translate_ageis_dataset_with_catagory_facebook_nllb-200-distilled-600M_categorized.log 2>&1 &
"""

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import nltk
import json
import math
import os
import sys

# ---------------------- Configuration ----------------------
USE_M2M = True  # True => NLLB-200; False => MarianMT (shhossain/opus-mt-en-to-bn)

# I/O paths
OUTPUT_DIR = "/home/tahad/ai-safety-bangla/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_M2M:
    output_translated = os.path.join(
        OUTPUT_DIR,
        "ageis_prompt_response_pairs_bangla_translation_facebook_nllb-200-distilled-600M_categorized.json",
    )
    filtered_output = os.path.join(
        OUTPUT_DIR,
        "ageis_prompt_response_pairs_facebook_nllb-200-distilled-600M_categorized.json",
    )
else:
    output_translated = os.path.join(
        OUTPUT_DIR,
        "ageis_prompt_response_pairs_bangla_translation_categorized.json",
    )
    filtered_output = os.path.join(
        OUTPUT_DIR,
        "ageis_prompt_response_pairs_categorized.json",
    )

BATCH_SIZE_PAIRS = 64     # number of prompt-response pairs per outer loop
SENT_MAX_LENGTH = 512     # max tokens for generation
PIPELINE_BATCH_SIZE = 32  # number of sentences sent per pipeline batch

# ---------------------- NLTK sentence splitter ----------------------
def _ensure_nltk_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    # Some newer NLTK builds need this as well:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

_ensure_nltk_punkt()

# ---------------------- Model / Pipeline Setup ----------------------
if USE_M2M:
    # facebook/nllb-200-distilled-600M (English -> Bangla)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Correct NLLB language codes
    SRC_CODE = "eng_Latn"
    TGT_CODE = "ben_Beng"

    # (Some versions require setting this too)
    tokenizer.src_lang = SRC_CODE

    # device=0 maps to the first *visible* GPU (e.g., phys GPU 3 if CUDA_VISIBLE_DEVICES=3)
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        device=0,
        src_lang=SRC_CODE,
        tgt_lang=TGT_CODE,
    )
else:
    # MarianMT (community model)
    model_name = "shhossain/opus-mt-en-to-bn"
    translator = pipeline("translation", model=model_name, device=0)
    SRC_CODE = None
    TGT_CODE = None

# ---------------------- Load & Filter Dataset ----------------------
print("Loading dataset: nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (split=train)")
dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")

# Keep rows where prompt_label exists and response_label == "safe".
# Store category_label = first violated category, or "safe" if none.
filtered_rows = []
for row in dataset:
    try:
        if row.get("prompt_label", None) is not None and row.get("response_label", None) == "safe":
            violated = row.get("violated_categories", None)
            if isinstance(violated, list) and len(violated) > 0:
                category = violated[0]  # pick the first violated category
            elif violated in [None, [], ""]:
                category = "safe"
            else:
                category = str(violated)

            filtered_rows.append({
                "prompt": row.get("prompt", "") or "",
                "response": row.get("response", "") or "",
                "prompt_label": row.get("prompt_label", None),
                "response_label": row.get("response_label", None),
                "category_label": category,
            })
    except Exception as e:
        # Skip malformed rows safely
        print(f"Skipping a row due to error: {e}", file=sys.stderr)

print(f"Filtered {len(filtered_rows)} rows")

# Save English filtered (pre-translation)
with open(filtered_output, "w", encoding="utf-8") as f:
    json.dump(filtered_rows, f, ensure_ascii=False, indent=2)
print(f"Saved English filtered dataset to {filtered_output}")

# ---------------------- Sentence batching helpers ----------------------
split_prompt_count = 0
split_response_count = 0

def split_sentences(text: str, is_prompt: bool = True):
    """Split text into sentences and increment counters if splitting occurs."""
    global split_prompt_count, split_response_count
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    sentences = nltk.sent_tokenize(text)
    if len(sentences) > 1:
        if is_prompt:
            split_prompt_count += 1
        else:
            split_response_count += 1
    return sentences if len(sentences) > 0 else [""]

def batch_translate_texts(texts, is_prompt=True, max_length=SENT_MAX_LENGTH):
    """
    Translate a list of texts by splitting into sentences, translating in a flat batch,
    and reassembling them back to the original texts.
    """
    if not texts:
        return texts

    batched_sentences = []
    indices = []  # map each sentence back to its original text index

    for idx, text in enumerate(texts):
        sents = split_sentences(text, is_prompt=is_prompt)
        for s in sents:
            s = s.strip()
            if s:
                batched_sentences.append(s)
                indices.append(idx)

    if not batched_sentences:
        return texts

    # Translate all sentences using the HF pipeline (let it internally batch)
    outputs = [""] * len(texts)
    translations = translator(
        batched_sentences,
        max_length=max_length,
        batch_size=PIPELINE_BATCH_SIZE,
        # Pass langs again to satisfy stricter Transformers versions (no-op for Marian)
        **({"src_lang": SRC_CODE, "tgt_lang": TGT_CODE} if SRC_CODE and TGT_CODE else {})
    )

    # Each item is like {"translation_text": "..."}
    for idx, trans in zip(indices, translations):
        translated_text = trans.get("translation_text", "").strip()
        if outputs[idx]:
            outputs[idx] += " " + translated_text
        else:
            outputs[idx] = translated_text

    # Fallback to original text if something ended empty (should be rare)
    for i in range(len(outputs)):
        if not outputs[i]:
            outputs[i] = texts[i]
    return outputs

# ---------------------- Translation Loop (streaming JSON) ----------------------
num_batches = math.ceil(len(filtered_rows) / BATCH_SIZE_PAIRS)
print(f"Translating in {num_batches} batches of {BATCH_SIZE_PAIRS} pairs...")

with open(output_translated, "w", encoding="utf-8") as f:
    f.write("[\n")  # begin JSON array

    for b in tqdm(range(num_batches), desc="Translating", unit="batch"):
        batch = filtered_rows[b * BATCH_SIZE_PAIRS : (b + 1) * BATCH_SIZE_PAIRS]

        prompts = [row["prompt"] for row in batch]
        responses = [row["response"] for row in batch]

        translated_prompts = batch_translate_texts(prompts, is_prompt=True)
        translated_responses = batch_translate_texts(responses, is_prompt=False)

        # Stitch back and write
        for i, (row, tp, tr) in enumerate(zip(batch, translated_prompts, translated_responses)):
            record = {
                "prompt": tp,
                "response": tr,
                "prompt_label": row["prompt_label"],
                "response_label": row["response_label"],
                "category_label": row["category_label"],
            }
            json.dump(record, f, ensure_ascii=False, indent=2)

            is_last_record = (b == num_batches - 1) and (i == len(batch) - 1)
            f.write("\n" if is_last_record else ",\n")

        f.flush()

    f.write("]\n")  # end JSON array

print(f"Saved translated rows to {output_translated}")
print(f"Prompts needing sentence splitting: {split_prompt_count}")
print(f"Responses needing sentence splitting: {split_response_count}")
