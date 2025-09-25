from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import nltk
import json
import math

# Setup
nltk.download("punkt")

# --- SWITCH HERE ---
USE_M2M = True   # set to False to use shhossain/opus-mt-en-to-bn instead

# --- MODEL SETUP ---
if USE_M2M:
    # facebook/m2m100_418M (English → Bangla)
    model_name = "facebook/m2m100_418M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang="en",
        tgt_lang="bn",
        device=0
    )
    output_translated = "/home/tahad/ai-safety-bangla/datasets/ageis_prompt_response_pairs_bangla_translation_m2m100.json"
    filtered_output = "/home/tahad/ai-safety-bangla/datasets/ageis_prompt_response_pairs_m2m100.json"
else:
    # shhossain/opus-mt-en-to-bn (MarianMT)
    model_name = "shhossain/opus-mt-en-to-bn"
    translator = pipeline("translation", model=model_name, device=0)
    output_translated = "/home/tahad/ai-safety-bangla/datasets/ageis_prompt_response_pairs_bangla_translation.json"
    filtered_output = "/home/tahad/ai-safety-bangla/datasets/ageis_prompt_response_pairs.json"

# Load dataset
dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")

# 🔎 Filtering: only keep rows where both labels exist AND response_label == "safe"
filtered_rows = [
    {
        "prompt": row["prompt"],
        "response": row["response"],
        "prompt_label": row["prompt_label"],
        "response_label": row["response_label"],
    }
    for row in dataset
    if row["prompt_label"] is not None and row["response_label"] == "safe"
]

print(f"Filtered {len(filtered_rows)} rows")

# --- Save English filtered dataset ---
with open(filtered_output, "w", encoding="utf-8") as f:
    json.dump(filtered_rows, f, ensure_ascii=False, indent=4)

print(f"Saved English filtered dataset to {filtered_output}")

# Counters
split_prompt_count, split_response_count = 0, 0

def split_sentences(text, is_prompt=True):
    """Split into sentences and update counters."""
    global split_prompt_count, split_response_count
    sentences = nltk.sent_tokenize(text)
    if len(sentences) > 1:
        if is_prompt:
            split_prompt_count += 1
        else:
            split_response_count += 1
    return sentences

def batch_translate_texts(texts, is_prompt=True, max_length=512):
    """Translate a list of texts in batch (sentence-level)."""
    batched_sentences = []
    indices = []  # map each sentence back to its text index

    for idx, text in enumerate(texts):
        sentences = split_sentences(text, is_prompt=is_prompt)
        for sent in sentences:
            if sent.strip():
                batched_sentences.append(sent)
                indices.append(idx)

    if not batched_sentences:
        return texts

    # Run translation in batch on GPU
    translations = translator(
        batched_sentences,
        max_length=max_length,
        batch_size=32,   # tuned for H100
    )

    # Reconstruct full texts
    outputs = [""] * len(texts)
    for idx, trans in zip(indices, translations):
        outputs[idx] += (" " + trans["translation_text"]).strip()

    return outputs

# Translation loop with batch processing + streaming JSON writing
batch_size = 128  # prompt-response pairs per loop
num_batches = math.ceil(len(filtered_rows) / batch_size)

with open(output_translated, "w", encoding="utf-8") as f:
    f.write("[\n")  # start JSON array

    for b in tqdm(range(num_batches), desc="Translating", unit="batch"):
        batch = filtered_rows[b*batch_size:(b+1)*batch_size]

        prompts = [row["prompt"] for row in batch]
        responses = [row["response"] for row in batch]

        translated_prompts = batch_translate_texts(prompts, is_prompt=True)
        translated_responses = batch_translate_texts(responses, is_prompt=False)

        results = []
        for row, tp, tr in zip(batch, translated_prompts, translated_responses):
            results.append({
                "prompt": tp,
                "response": tr,
                "prompt_label": row["prompt_label"],
                "response_label": row["response_label"],
            })

        # Write each batch immediately
        for i, r in enumerate(results):
            json.dump(r, f, ensure_ascii=False, indent=4)
            if not (b == num_batches-1 and i == len(results)-1):
                f.write(",\n")
            else:
                f.write("\n")

        f.flush()  # ensure batch is written to disk

    f.write("]\n")  # close JSON array

print(f"Saved translated rows to {output_translated}")
print(f"Prompts needing sentence splitting: {split_prompt_count}")
print(f"Responses needing sentence splitting: {split_response_count}")

# Run with:
# CUDA_VISIBLE_DEVICES=3 nohup python translate_ageis_dataset.py > /home/tahad/ai-safety-bangla/logs/translate_ageis_dataset.log 2>&1 &
