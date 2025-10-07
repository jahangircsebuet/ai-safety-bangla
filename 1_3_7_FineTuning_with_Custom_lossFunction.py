#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TigerLLM Safety Fine-Tuning (LoRA + Contrastive Safety Loss)
------------------------------------------------------------
✓ Uses exact model loader (T5Config + LoRA)
✓ Combines CE + margin-based contrastive safety loss
✓ Unsafe prompts:
    → if safe response exists: learn redirection
    → else: alternate refusal & unsafe response for contrastive loss
✓ Safe prompts: normal completion
✓ AMP + GradScaler + Gradient Accumulation
✓ Logs and saves per-epoch metrics
"""

import os, json, math, random, warnings
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# ===============================================================
# Config
# ===============================================================
BASE_MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
RESULTS_BASE_DIR = "./FineTuned_Models"
LOSS_TYPE = "_LoRA_CE+Contrastive"
SAVE_FOLDER_NAME = BASE_MODEL_NAME.split("/")[-1] + LOSS_TYPE

EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM = 8
LR = 5e-5
FP16 = True
MAX_LEN = 512
GAMMA = 9.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=FutureWarning)

REFUSAL_TEMPLATE = (
    "এই প্রশ্নটি বিপজ্জনক। এর উত্তরে অসুরক্ষিত বিষয় হতে পারে। "
    "দয়া করে নিরাপদ প্রথা অনুসরণ করুন।"
)

# ===============================================================
# AMP Setup
# ===============================================================
try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_ARGS = {"device_type": "cuda"}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AUTOCAST_ARGS = {"arg": "cuda"}

scaler = GradScaler(enabled=(FP16 and DEVICE.startswith("cuda")))

# ===============================================================
# Load Model / Tokenizer
# ===============================================================
print(f"Loading {BASE_MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
config = T5Config.from_pretrained(BASE_MODEL_NAME)

if config.decoder_start_token_id is None:
    config.decoder_start_token_id = tokenizer.pad_token_id

base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, config=config)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q", "v"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config).to(DEVICE)

# ===============================================================
# Dataset Loading
# ===============================================================
dataset_paths = [
    "/home/tahad/ai-safety-bangla/datasets/LLamaGuard_prompt_response_pairs_bangla_translation_categorized.json",
    "/home/tahad/ai-safety-bangla/datasets/ageis_prompt_response_pairs_bangla_translation_facebook_nllb-200-distilled-600M_categorized.json",
    "/home/tahad/ai-safety-bangla/datasets/Ageis_unsafe_responses_translated.json",
]

raw_items = []
for path in dataset_paths:
    with open(path, "r", encoding="utf-8") as f:
        raw_items.extend(json.load(f))

# --- Modified Encode Function ---
def encode_entry(entry):
    if entry.get("prompt_label") == "unsafe":
        if entry.get("response_label") == "safe":
            # Safe redirect exists → teach the model redirection
            response = entry["response"]
        else:
            # No safe response → alternate between refusal & unsafe for contrastive training
            response = REFUSAL_TEMPLATE if random.random() < 0.5 else entry["response"]
    else:
        # Safe prompt → normal supervised learning
        response = entry["response"]

    return {
        "input_text": entry["prompt"],
        "output_text": response,
        "prompt_label": entry["prompt_label"],
        "response_label": entry.get("response_label", "safe"),
        "category_label": entry.get("category_label", "safe"),
    }

encoded = [encode_entry(e) for e in raw_items]

random.shuffle(encoded)

dataset = Dataset.from_dict({
    "input_text": [e["input_text"] for e in encoded],
    "output_text": [e["output_text"] for e in encoded],
    "prompt_label": [e["prompt_label"] for e in encoded],
    "response_label": [e["response_label"] for e in encoded],
    "category_label": [e["category_label"] for e in encoded],
})

# Split train/val/test
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
temp_dataset = split["test"]
val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

# Save raw test split
save_root = os.path.join(RESULTS_BASE_DIR, SAVE_FOLDER_NAME)
model_dir = os.path.join(save_root, "model")
results_dir = os.path.join(save_root, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "test_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(test_dataset.to_dict(), f, ensure_ascii=False, indent=2)
print(f"Saved raw test split: {results_dir}/test_dataset.json")

# ===============================================================
# Log-likelihood calculator
# ===============================================================
def batch_loglikelihood(prompts, targets, model, tokenizer, max_len=512, device="cuda", batch_size=4):
    all_scores = []
    for i in range(0, len(prompts), batch_size):
        bp, bt = prompts[i:i+batch_size], targets[i:i+batch_size]
        enc = tokenizer(bp, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with tokenizer.as_target_tokenizer():
            labs = tokenizer(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        labels = labs["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            logits = out.logits
        log_probs = F.log_softmax(logits, dim=-1)
        shift_labels = labels[:, 1:].clone()
        shift_logits = log_probs[:, :-1, :]
        valid_mask = (shift_labels != -100)
        shift_labels[~valid_mask] = 0
        tok_logp = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        tok_logp = tok_logp * valid_mask.float()
        seq_scores = tok_logp.sum(dim=1)
        all_scores.append(seq_scores)
    return torch.cat(all_scores, dim=0)

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
    optimizer.zero_grad()

    ce_loss_epoch, con_loss_epoch = [], []

    for step in tqdm(range(steps_per_epoch), desc="Training"):
        batch = train_dataset.shuffle(seed=epoch*100+step).select(range(BATCH_SIZE)).to_dict()
        prompts, refs, labels_prompt = batch["input_text"], batch["output_text"], batch["prompt_label"]

        # Cross-Entropy loss
        enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LEN).to(DEVICE)
        with tokenizer.as_target_tokenizer():
            labs = tokenizer(refs, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LEN).to(DEVICE)
        labels = labs["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        with autocast(**AUTOCAST_ARGS, enabled=(FP16 and DEVICE.startswith("cuda"))):
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            ce_loss = out.loss

        # Contrastive loss for unsafe prompts
        unsafe_indices = [i for i, pl in enumerate(labels_prompt) if pl == "unsafe"]
        contrastive_loss = 0.0
        if unsafe_indices:
            prompts_u = [prompts[i] for i in unsafe_indices]
            y_pos = [REFUSAL_TEMPLATE for _ in unsafe_indices]
            y_neg = [refs[i] for i in unsafe_indices]

            s_pos = batch_loglikelihood(prompts_u, y_pos, model, tokenizer, device=DEVICE)
            s_neg = batch_loglikelihood(prompts_u, y_neg, model, tokenizer, device=DEVICE)
            margin_loss = torch.clamp(GAMMA - (s_pos - s_neg), min=0).mean()
            contrastive_loss = margin_loss

        total_loss = ce_loss + contrastive_loss
        ce_loss_epoch.append(ce_loss.item())
        con_loss_epoch.append(float(contrastive_loss))

        # Backprop
        scaler.scale(total_loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Epoch stats
    ce_epoch_mean = sum(ce_loss_epoch)/len(ce_loss_epoch)
    con_epoch_mean = sum(con_loss_epoch)/len(con_loss_epoch)
    ce_history.append(ce_epoch_mean)
    con_history.append(con_epoch_mean)
    total_history.append(ce_epoch_mean + con_epoch_mean)
    print(f"Epoch {epoch+1}: CE={ce_epoch_mean:.4f}, Contrastive={con_epoch_mean:.4f}")

    # Save checkpoint
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Checkpoint saved to {model_dir}")

# ===============================================================
# Plot training curves
# ===============================================================
plt.figure(figsize=(6,4))
plt.plot(ce_history, label="Cross-Entropy")
plt.plot(con_history, label="Contrastive")
plt.plot(total_history, label="Total")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_loss_curve.png"), dpi=300)
plt.show()

print("Training complete.")
