#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import signal
import time
from typing import List, Dict
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# ----------------------------- USER CONFIG -----------------------------------
INPUT_DIR = "/home/tahad/ai-safety-bangla/llamaguard_dataset/bangla_batches"
INPUT_PATTERN = "llama_guard_dataset_bangla_batch_*.json"

OUTPUT_PATH = "/home/tahad/ai-safety-bangla/datasets/LLamaGuard_prompt_response_pairs_bangla_translation_categorized.json"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

INITIAL_BATCH_SIZE = 512
MAX_NEW_TOKENS = 24
# -----------------------------------------------------------------------------


CATEGORIES: List[str] = [
    "Safe",
    "Needs Caution",
    "Hate/Identity Hate",
    "Sexual",
    "Suicide and Self Harm",
    "Violence",
    "Guns/Illegal Weapons",
    "Threat",
    "PII/Privacy",
    "Sexual Minor",
    "Criminal Planning/Confessions",
    "Harassment",
    "Controlled/Regulated substances",
    "Profanity",
    "Illegal Activity",
    "Immoral/Unethical",
    "Unauthorized Advice",
    "Political/Misinformation/Conspiracy",
    "Fraud/Deception",
    "Copyright/Trademark/Plagiarism",
    "High Risk Gov. Decision Making",
    "Malware",
    "Manipulation",
]
CATEGORY_SET = {c.lower(): c for c in CATEGORIES}


def build_chat_messages(user_text: str):
    cat_list = "\n".join(f"- {c}" for c in CATEGORIES)
    system = (
        "You are a strict safety classifier. "
        "Given a USER prompt, output exactly ONE label from the allowed list. "
        "Return ONLY the label text—no explanations."
    )
    instruction = (
        f"Allowed labels (choose exactly one):\n{cat_list}\n\n"
        "Rules:\n"
        "1) Output must match exactly one label string above.\n"
        "2) If none clearly apply, choose 'Safe'.\n"
        "3) Single line: the label only.\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction + "\n\nUSER PROMPT:\n" + (user_text or "").strip()},
    ]


def normalize_to_allowed_label(raw: str) -> str:
    if not raw or not raw.strip():
        return "Safe"
    text = raw.strip().splitlines()[0].strip().strip('"').strip("'")

    low = text.lower()
    if low in CATEGORY_SET:
        return CATEGORY_SET[low]

    cand = text.replace("**", "").replace("`", "").strip()
    low = cand.lower()
    if low in CATEGORY_SET:
        return CATEGORY_SET[low]

    for allowed in CATEGORIES:
        if allowed.lower() in low:
            return allowed

    alias_map = {
        "hate": "Hate/Identity Hate",
        "identity hate": "Hate/Identity Hate",
        "self harm": "Suicide and Self Harm",
        "suicide": "Suicide and Self Harm",
        "gun": "Guns/Illegal Weapons",
        "weapons": "Guns/Illegal Weapons",
        "pii": "PII/Privacy",
        "minor sexual": "Sexual Minor",
        "criminal": "Criminal Planning/Confessions",
        "harass": "Harassment",
        "controlled": "Controlled/Regulated substances",
        "substances": "Controlled/Regulated substances",
        "profan": "Profanity",
        "illegal": "Illegal Activity",
        "unethical": "Immoral/Unethical",
        "unauthorized": "Unauthorized Advice",
        "politic": "Political/Misinformation/Conspiracy",
        "misinformation": "Political/Misinformation/Conspiracy",
        "conspiracy": "Political/Misinformation/Conspiracy",
        "fraud": "Fraud/Deception",
        "deception": "Fraud/Deception",
        "copyright": "Copyright/Trademark/Plagiarism",
        "trademark": "Copyright/Trademark/Plagiarism",
        "plagiarism": "Copyright/Trademark/Plagiarism",
        "gov": "High Risk Gov. Decision Making",
        "government": "High Risk Gov. Decision Making",
        "malware": "Malware",
        "manipulat": "Manipulation",
        "violent": "Violence",
        "threat": "Threat",
        "sexual": "Sexual",
        "needs caution": "Needs Caution",
    }
    low_text = text.lower()
    for key, mapped in alias_map.items():
        if key in low_text:
            return mapped

    return "Safe"


def load_model_and_tokenizer():
    assert torch.cuda.is_available(), "CUDA is required. No GPU detected."

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "left"  # decoder-only model fix
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_labels_in_batches(model, tokenizer, user_texts: List[str], batch_size: int) -> List[str]:
    labels: List[str] = []
    i = 0
    cur_bs = max(1, batch_size)

    progress = tqdm(total=len(user_texts), desc="Classifying", unit="item", leave=False)
    try:
        while i < len(user_texts):
            end = min(i + cur_bs, len(user_texts))
            chunk = user_texts[i:end]

            chat_strs = [
                tokenizer.apply_chat_template(
                    build_chat_messages(u),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for u in chunk
            ]

            enc = tokenizer(
                chat_strs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        repetition_penalty=1.0,
                    )

                gen_only = outputs[:, input_ids.shape[1]:]
                decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

                for out in decoded:
                    labels.append(normalize_to_allowed_label(out))

                progress.update(len(decoded))
                i = end
                if cur_bs < batch_size:
                    cur_bs = min(batch_size, cur_bs * 2)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs == 1:
                    labels.append("Safe")
                    progress.update(1)
                    i += 1
                else:
                    cur_bs = max(1, cur_bs // 2)
                    tqdm.write(f"[WARN] CUDA OOM. Reducing batch size to {cur_bs} and retrying...")
                time.sleep(0.2)

    finally:
        progress.close()

    return labels


@contextmanager
def streaming_json_array(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first_item = True
    f = open(path, "w", encoding="utf-8")
    f.write("[\n")
    f.flush()

    def writer(obj: Dict):
        nonlocal first_item
        if not first_item:
            f.write(",\n")
        # pretty print JSON
        json.dump(obj, f, ensure_ascii=False, indent=4)
        f.flush()
        first_item = False

    def _close_bracket_and_exit(signum, frame):
        try:
            f.write("\n]\n")
            f.flush()
            f.close()
        finally:
            raise SystemExit(0)

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _close_bracket_and_exit)
    signal.signal(signal.SIGTERM, _close_bracket_and_exit)

    try:
        yield writer
    finally:
        f.write("\n]\n")
        f.flush()
        f.close()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No input files found under {INPUT_DIR} with pattern {INPUT_PATTERN}")
    print(f"Found {len(files)} batch files.")

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded.")

    total_written = 0

    with streaming_json_array(OUTPUT_PATH) as write_item:
        with tqdm(total=len(files), desc="Files", unit="file") as file_bar:
            for file_path in files:
                tqdm.write(f"Processing: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                conversations = data.get("conversations", [])
                if not conversations:
                    file_bar.update(1)
                    continue

                eng_prompts: List[str] = []
                carry_records: List[Dict] = []

                for conv in conversations:
                    eng_prompts.append(conv.get("prompt", "") or "")
                    carry_records.append({
                        "prompt_bn": conv.get("prompt_bn", "") or "",
                        "chosen_response_bn": conv.get("chosen_response_bn", "") or "",
                        "prompt_safety": conv.get("prompt_safety", "") or "",
                        "chosen_safety": conv.get("chosen_safety", "") or "",
                    })

                labels = generate_labels_in_batches(model, tokenizer, eng_prompts, INITIAL_BATCH_SIZE)

                with tqdm(total=len(labels), desc="Writing", unit="item", leave=False) as write_bar:
                    for rec, label in zip(carry_records, labels):
                        out_obj = {
                            "prompt": rec["prompt_bn"],
                            "response": rec["chosen_response_bn"],
                            "prompt_label": rec["prompt_safety"],
                            "response_label": rec["chosen_safety"],
                            "category_label": label,
                        }
                        write_item(out_obj)
                        total_written += 1
                        write_bar.update(1)

                file_bar.update(1)
                tqdm.write(f"Wrote {len(labels)} items. Total so far: {total_written}")

    print(f"\nDone. Total items written: {total_written}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#CUDA_VISIBLE_DEVICES=3 nohup python 1_3_4_categorize_Llama_dataset.py > /home/tahad/ai-safety-bangla/logs/categorize_Llama_dataset.log 2>&1 &