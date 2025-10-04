#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Fine-Tuning for TigerLLM-1B-it using LoRA (Cross-Entropy Only)
----------------------------------------------------------------------
This version mirrors the architecture and directory structure used in
the contrastive fine-tuning pipeline, allowing direct comparison.

Model: md-nishat-008/TigerLLM-1B-it
Dataset: AI-safety Bangla prompt-response pairs
"""

import os, json, math, warnings
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Config,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# ===============================================================
# CONFIGURATION
# ===============================================================
BASE_MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
RESULTS_BASE_DIR = "./FineTuned_Models"
LOSS_TYPE = "_LoRA_CE"  # for evaluation compatibility
SAVE_FOLDER_NAME = BASE_MODEL_NAME.split("/")[-1] + LOSS_TYPE

EPOCHS = 3
BATCH_SIZE = 128
GRAD_ACCUM = 8
LR = 5e-5
FP16 = True
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"🚀 Using device: {DEVICE}")

# ===============================================================
# LOAD MODEL + TOKENIZER
# ===============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
config = T5Config.from_pretrained(BASE_MODEL_NAME)
if config.decoder_start_token_id is None:
    config.decoder_start_token_id = tokenizer.pad_token_id

base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME, config=config)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(base_model, lora_config).to(DEVICE)

# ===============================================================
# LOAD DATASET
# ===============================================================
dataset_paths = [
    "/home/tahad/ai-safety-bangla/datasets/LLamaGuard_prompt_response_pairs_bangla_translation_categorized.json",
    "/home/tahad/ai-safety-bangla/datasets/ageis_prompt_response_pairs_bangla_translation_facebook_nllb-200-distilled-600M_categorized.json",
]

raw_items = []
for path in dataset_paths:
    with open(path, "r", encoding="utf-8") as f:
        raw_items.extend(json.load(f))

REFUSAL_TEMPLATE = (
    "এই প্রশ্নটি বিপজ্জনক। এর উত্তরে অসুরক্ষিত বিষয় হতে পারে। "
    "দয়া করে নিরাপদ প্রথা অনুসরণ করুন।"
)

def encode_entry(entry):
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
        "prompt_label": entry["prompt_label"],
        "category_label": entry.get("category_label", "safe")
    }

encoded = [encode_entry(e) for e in raw_items]
dataset = Dataset.from_dict({
    "input_text": [e["input_text"] for e in encoded],
    "output_text": [e["output_text"] for e in encoded],
    "prompt_label": [e["prompt_label"] for e in encoded],
    "category_label": [e["category_label"] for e in encoded],
})

# ===============================================================
# SPLIT DATASET (Train / Val / Test)
# ===============================================================
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
temp_dataset = split["test"]
val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

# Save test split for evaluation
save_root = os.path.join(RESULTS_BASE_DIR, SAVE_FOLDER_NAME)
model_dir = os.path.join(save_root, "model")
results_dir = os.path.join(save_root, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

test_path = os.path.join(results_dir, "test_dataset.json")
with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_dataset.to_dict(), f, ensure_ascii=False, indent=2)
print(f"📁 Saved test dataset to: {test_path}")

# ===============================================================
# TOKENIZATION
# ===============================================================
def tokenize_fn(batch):
    model_inputs = tokenizer(batch["input_text"], padding="max_length",
                             truncation=True, max_length=MAX_LEN)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["output_text"], padding="max_length",
                           truncation=True, max_length=MAX_LEN)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ===============================================================
# TRAINING LOOP (Cross-Entropy Only)
# ===============================================================
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=os.path.join(results_dir, "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=FP16,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"\n=== Training LoRA Baseline ({LOSS_TYPE}) ===")
trainer.train()

# ===============================================================
# SAVE MODEL + LOSS CURVES
# ===============================================================
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"✅ Model + tokenizer saved to {model_dir}")

# Training curves
history = trainer.state.log_history
train_loss = [x["loss"] for x in history if "loss" in x]
eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

plt.figure(figsize=(8,6))
if train_loss: plt.plot(train_loss, label="Train Loss")
if eval_loss: plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Steps"); plt.ylabel("Loss")
plt.title("Training vs Validation Loss (LoRA Baseline)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=300)
print(f"📊 Loss curve saved to {results_dir}")

print("🎯 LoRA baseline training complete.")
