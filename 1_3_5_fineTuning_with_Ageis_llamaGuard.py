#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CE-Only Fine-Tuning for TigerLLM-1B-it (Gemma3 Causal LM) using LoRA
-------------------------------------------------------------------
This script performs a baseline fine-tuning of the TigerLLM-1B-it model
on a set of prompt/response pairs using a cross-entropy loss alone.
It uses the correct Gemma3 causal LM architecture and LoRA adapters.

✓ Uses AutoModelForCausalLM and AutoConfig (Gemma3)
✓ LoRA on q_proj/k_proj/v_proj/o_proj
✓ Concatenates prompt + response, masks prompt tokens in labels
✓ Supports AMP (BF16/FP16) + gradient accumulation
✓ Loads train/val/test from fixed JSON paths
✓ Saves checkpoints and plots CE loss curve
"""

import json
import math
import os
import random
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model

# ===============================================================
# CONFIGURATION
# ===============================================================
BASE_MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
RESULTS_BASE_DIR = "./FineTuned_Models"
LOSS_TYPE = "_LoRA_CE"
SAVE_FOLDER_NAME = BASE_MODEL_NAME.split("/")[-1] + LOSS_TYPE

EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM = 8
LR = 5e-5

# Use BF16 if available, else FP16 for mixed precision
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = (not USE_BF16)

# Sequence length settings
MAX_SOURCE_LEN = 2048
MAX_TARGET_LEN = 512
MAX_TOTAL_LEN = MAX_SOURCE_LEN + MAX_TARGET_LEN

SEED = 42
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

REFUSAL_TEMPLATE = (
    "এই প্রশ্নটি বিপজ্জনক। এর উত্তরে অসুরক্ষিত বিষয় হতে পারে। "
    "দয়া করে নিরাপদ প্রথা অনুসরণ করুন।"
)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===============================================================
# LOAD MODEL + TOKENIZER
# ===============================================================
print(f"Loading {BASE_MODEL_NAME} (Gemma3 causal LM) ...")

config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
config.use_cache = False
config.attn_implementation = "sdpa"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    config=config,
    torch_dtype=(torch.bfloat16 if USE_BF16 else None),
)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config).to(DEVICE)
model.train()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: total={total_params/1e6:.1f}M, trainable={trainable_params/1e6:.1f}M")

# ===============================================================
# LOAD DATASETS (Train / Val / Test)
# ===============================================================
train_paths = [
    "/home/tahad/ai-safety-bangla/Final_dataset/train.json",
    #"/home/tahad/ai-safety-bangla/Final_dataset/Unsafe_responses.json",
]
VAL_DATA_PATH = "/home/tahad/ai-safety-bangla/Final_dataset/val.json"
TEST_DATA_PATH = "/home/tahad/ai-safety-bangla/Final_dataset/test.json"

raw_train_items = []
for path in train_paths:
    with open(path, "r", encoding="utf-8") as f:
        raw_train_items.extend(json.load(f))

if not raw_train_items:
    raise ValueError("No training data found — check train_paths!")

# Merge + shuffle once for balanced order
random.shuffle(raw_train_items)

def encode_entry(entry: Dict) -> Dict:
    if entry.get("prompt_label") == "unsafe":
        if entry.get("response_label") == "safe":
            response = entry["response"]
        else:
            response = REFUSAL_TEMPLATE
    else:
        response = entry["response"]
    return {
        "input_text": entry["prompt"],
        "output_text": response,
        "prompt_label": entry.get("prompt_label", "safe"),
        "prompt_category": entry.get("prompt_category", "safe"),
    }

encoded_train = [encode_entry(e) for e in raw_train_items]
train_dataset = Dataset.from_dict({
    "input_text": [e["input_text"] for e in encoded_train],
    "output_text": [e["output_text"] for e in encoded_train],
    "prompt_label": [e["prompt_label"] for e in encoded_train],
    "prompt_category": [e["prompt_category"] for e in encoded_train],
})

# Load validation and test datasets
with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Save val/test datasets for evaluation
save_root = os.path.join(RESULTS_BASE_DIR, SAVE_FOLDER_NAME)
model_dir = os.path.join(save_root, "model")
results_dir = os.path.join(save_root, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "val_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)
with open(os.path.join(results_dir, "test_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Loaded Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ===============================================================
# BUILD CAUSAL BATCHES
# ===============================================================
def build_causal_batch(prompts, targets, tokenizer, max_source_len, max_target_len, device):
    enc_p = tokenizer(prompts, padding=True, truncation=True, max_length=max_source_len, return_tensors="pt")
    enc_t = tokenizer(targets, padding=True, truncation=True, max_length=max_target_len, add_special_tokens=False, return_tensors="pt")

    input_ids_list, attn_list, labels_list = [], [], []
    pad_id = tokenizer.pad_token_id

    for i in range(len(prompts)):
        p_ids = enc_p.input_ids[i]
        t_ids = enc_t.input_ids[i]
        if tokenizer.eos_token_id is not None and (len(t_ids) == 0 or t_ids[-1] != tokenizer.eos_token_id):
            t_ids = torch.cat([t_ids, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)])
        combined = torch.cat([p_ids, t_ids])[:MAX_TOTAL_LEN]
        p_len = min(len(p_ids), len(combined))
        attn = torch.ones_like(combined)
        labels = combined.clone()
        if p_len > 0:
            labels[:p_len] = -100
        input_ids_list.append(combined)
        attn_list.append(attn)
        labels_list.append(labels)

    max_len = max(x.size(0) for x in input_ids_list)

    def _pad(seq_list, value: int):
        out = torch.full((len(seq_list), max_len), fill_value=value, dtype=torch.long)
        for i, s in enumerate(seq_list):
            out[i, : s.size(0)] = s
        return out

    input_ids = _pad(input_ids_list, pad_id).to(device)
    attention_mask = _pad(attn_list, 0).to(device)
    labels = _pad(labels_list, -100).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ===============================================================
# TRAINING LOOP (Cross-Entropy Only)
# ===============================================================
from torch.cuda.amp import autocast, GradScaler
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
scaler = GradScaler(enabled=torch.cuda.is_available())

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM))
ce_history = []

print(f"=== Training CE-only LoRA baseline for {EPOCHS} epochs ({steps_per_epoch} steps/epoch) ===")

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_losses = []

    for step in tqdm(range(steps_per_epoch), desc="Training"):
        batch = train_dataset.shuffle(seed=epoch * 100 + step).select(range(BATCH_SIZE)).to_dict()
        prompts, targets = batch["input_text"], batch["output_text"]
        batch_data = build_causal_batch(prompts, targets, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, DEVICE)
        with autocast(enabled=torch.cuda.is_available(), dtype=AUTOCAST_DTYPE):
            out = model(**batch_data)
            loss = out.loss
        epoch_losses.append(loss.item())
        scaler.scale(loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    mean_loss = sum(epoch_losses) / len(epoch_losses)
    ce_history.append(mean_loss)
    print(f"Epoch {epoch+1} average CE loss: {mean_loss:.4f}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Checkpoint saved to {model_dir}")

# ===============================================================
# SAVE MODEL + LOSS CURVE
# ===============================================================
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Final model + tokenizer saved to {model_dir}")

plt.figure(figsize=(6,4))
plt.plot(ce_history, label="Cross-Entropy Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("CE Loss over Epochs (LoRA Baseline)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ce_loss_curve.png"), dpi=300)
plt.close()
print(f"Loss curve saved to {results_dir}/ce_loss_curve.png")

print("Training complete.")
