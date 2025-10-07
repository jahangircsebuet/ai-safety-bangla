#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TigerLLM-1B-it Safety Fine-Tuning (LoRA + CE + Contrastive) — Gemma3 Causal LM
------------------------------------------------------------------------------
This version:
✓ Uses the final dataset paths for train/val/test (no random splitting)
✓ Loads TigerLLM-1B-it correctly as a Gemma3 causal LM
✓ Trains with combined Cross-Entropy + Contrastive loss
✓ Saves checkpoints and plots per-epoch CE / Contrastive / Total losses
"""

import os
import json
import math
import random
import warnings
from typing import List, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# ===============================================================
# CONFIGURATION
# ===============================================================
BASE_MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"  # Gemma3 causal LM
RESULTS_BASE_DIR = "./FineTuned_Models"
LOSS_TYPE = "_LoRA_CE+Contrastive"
SAVE_FOLDER_NAME = BASE_MODEL_NAME.split("/")[-1] + LOSS_TYPE

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM = 8
LR = 5e-5

# Precision setup
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = not USE_BF16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Length settings
MAX_SOURCE_LEN = 2048
MAX_TARGET_LEN = 512
MAX_TOTAL_LEN = MAX_SOURCE_LEN + MAX_TARGET_LEN

# Contrastive loss margin
GAMMA = 9.0
SEED = 42

warnings.filterwarnings("ignore", category=FutureWarning)

REFUSAL_TEMPLATE = (
    "এই প্রশ্নটি বিপজ্জনক। এর উত্তরে অসুরক্ষিত বিষয় হতে পারে। "
    "দয়া করে নিরাপদ প্রথা অনুসরণ করুন।"
)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===============================================================
# AMP Setup
# ===============================================================
try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_ARGS = {
        "device_type": "cuda",
        "dtype": torch.bfloat16 if USE_BF16 else torch.float16,
    }
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    AUTOCAST_ARGS = {"dtype": torch.float16}

scaler = GradScaler(enabled=torch.cuda.is_available())

# ===============================================================
# Load Model / Tokenizer
# ===============================================================
print(f"Loading {BASE_MODEL_NAME} (Gemma3 Causal LM) ...")

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
# Dataset Loading (No Random Splits)
# ===============================================================
train_paths = [
    "/home/tahad/ai-safety-bangla/Final_dataset/train.json",
    "/home/tahad/ai-safety-bangla/Final_dataset/Unsafe_responses.json"
]
VAL_DATA_PATH = "/home/tahad/ai-safety-bangla/Final_dataset/val.json"
TEST_DATA_PATH = "/home/tahad/ai-safety-bangla/Final_dataset/test.json"

raw_train_items = []
for path in train_paths:
    with open(path, "r", encoding="utf-8") as f:
        raw_train_items.extend(json.load(f))

random.shuffle(raw_train_items)

with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

def encode_entry(entry: Dict) -> Dict:
    prompt = entry["prompt"]
    if entry.get("prompt_label") == "unsafe":
        if entry.get("response_label") == "safe":
            response = entry["response"]
        else:
            response = REFUSAL_TEMPLATE if random.random() < 0.5 else entry["response"]
    else:
        response = entry["response"]

    return {
        "input_text": prompt,
        "output_text": response,
        "prompt_label": entry.get("prompt_label", "safe"),
        "response_label": entry.get("response_label", "safe"),
        "prompt_category": entry.get("prompt_category", "safe"),
    }

encoded_train = [encode_entry(e) for e in raw_train_items]
train_dataset = Dataset.from_dict({
    "input_text": [e["input_text"] for e in encoded_train],
    "output_text": [e["output_text"] for e in encoded_train],
    "prompt_label": [e["prompt_label"] for e in encoded_train],
    "response_label": [e["response_label"] for e in encoded_train],
    "prompt_category": [e["prompt_category"] for e in encoded_train],
})

val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

save_root = os.path.join(RESULTS_BASE_DIR, SAVE_FOLDER_NAME)
model_dir = os.path.join(save_root, "model")
results_dir = os.path.join(save_root, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "val_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)
with open(os.path.join(results_dir, "test_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
print(f"Loaded Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")

# ===============================================================
# Utilities
# ===============================================================
def build_causal_batch(prompts, targets, tokenizer, max_source_len, max_target_len, device):
    enc_p = tokenizer(prompts, padding=True, truncation=True, max_length=max_source_len, return_tensors="pt")
    enc_t = tokenizer(targets, padding=True, truncation=True, max_length=max_target_len,
                      add_special_tokens=False, return_tensors="pt")

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
    def _pad(seq_list, value: int) -> torch.Tensor:
        out = torch.full((len(seq_list), max_len), fill_value=value, dtype=torch.long)
        for i, s in enumerate(seq_list):
            out[i, : s.size(0)] = s
        return out

    input_ids = _pad(input_ids_list, pad_id).to(device)
    attention_mask = _pad(attn_list, 0).to(device)
    labels = _pad(labels_list, -100).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@torch.no_grad()
def batch_loglikelihood_causal(prompts, targets, model, tokenizer, device, max_source_len, max_target_len, batch_size=4):
    model.eval()
    scores = []
    for i in range(0, len(prompts), batch_size):
        bp = prompts[i:i+batch_size]
        bt = targets[i:i+batch_size]
        batch = build_causal_batch(bp, bt, tokenizer, max_source_len, max_target_len, device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        labels = batch["labels"][:, 1:]
        valid = (labels != -100)
        labels_clipped = labels.clone()
        labels_clipped[~valid] = 0
        token_logp = torch.gather(log_probs, -1, labels_clipped.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid.float()
        seq_scores = token_logp.sum(dim=1)
        scores.append(seq_scores.cpu())
    model.train()
    return torch.cat(scores, dim=0).to(device)

# ===============================================================
# Training Loop
# ===============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM))
ce_history, con_history, total_history = [], [], []

print(f"=== Training for {EPOCHS} epochs, {steps_per_epoch} steps/epoch ===")

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ce_loss_epoch, con_loss_epoch = [], []

    for step in tqdm(range(steps_per_epoch), desc="Training"):
        batch = train_dataset.shuffle(seed=epoch*100+step).select(range(BATCH_SIZE)).to_dict()
        prompts, targets, plabels = batch["input_text"], batch["output_text"], batch["prompt_label"]

        causal_batch = build_causal_batch(prompts, targets, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, DEVICE)
        with autocast(**AUTOCAST_ARGS, enabled=torch.cuda.is_available()):
            out = model(input_ids=causal_batch["input_ids"], attention_mask=causal_batch["attention_mask"], labels=causal_batch["labels"])
            ce_loss = out.loss

        # Contrastive Loss
        unsafe_idx = [i for i, pl in enumerate(plabels) if pl == "unsafe"]
        contrastive_loss = torch.tensor(0.0, device=DEVICE)
        if unsafe_idx:
            prompts_u = [prompts[i] for i in unsafe_idx]
            pos_targets = [REFUSAL_TEMPLATE for _ in unsafe_idx]
            neg_targets = [targets[i] for i in unsafe_idx]
            s_pos = batch_loglikelihood_causal(prompts_u, pos_targets, model, tokenizer, DEVICE, MAX_SOURCE_LEN, MAX_TARGET_LEN)
            s_neg = batch_loglikelihood_causal(prompts_u, neg_targets, model, tokenizer, DEVICE, MAX_SOURCE_LEN, MAX_TARGET_LEN)
            margin = GAMMA - (s_pos - s_neg)
            contrastive_loss = torch.clamp(margin, min=0).mean()

        total_loss = ce_loss + contrastive_loss
        ce_loss_epoch.append(ce_loss.item())
        con_loss_epoch.append(float(contrastive_loss.item()))

        scaler.scale(total_loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    ce_mean = sum(ce_loss_epoch) / len(ce_loss_epoch)
    con_mean = sum(con_loss_epoch) / len(con_loss_epoch)
    ce_history.append(ce_mean)
    con_history.append(con_mean)
    total_history.append(ce_mean + con_mean)
    print(f"Epoch {epoch+1}: CE={ce_mean:.4f}, Contrastive={con_mean:.4f}")

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Checkpoint saved to {model_dir}")

# ===============================================================
# Plot Training Curves
# ===============================================================
plt.figure(figsize=(7, 4))
plt.plot(ce_history, label="Cross-Entropy")
plt.plot(con_history, label="Contrastive")
plt.plot(total_history, label="Total")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Components (Gemma3 Causal LM + LoRA)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_loss_curve.png"), dpi=300)
plt.close()

print(" Training complete.")
