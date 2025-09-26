# -*- coding: utf-8 -*-
import os
import json
import warnings
from typing import List

# Assume you've already run:  huggingface-cli login
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence tokenizers fork warning

# ============ Libs ============
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import transformers

# quieten some noisy warns
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")

print(transformers.__file__)
print(transformers.__version__)  # should print 4.56.2 per your env

# ============ Paths ============
PROJECT_ROOT = "/home/jnewson/projects/ai-safety-bangla"
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "fullft-bangla-llm", "titulm-llama-3.2-1b-v1.1-fullfinetune")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, "finetuning_dataset_llamaguard_bangla.json")
TRAIN_SPLIT  = os.path.join(DATA_DIR, "finetune_banglaguard_train.json")
TEST_SPLIT   = os.path.join(DATA_DIR, "finetune_banglaguard_test.json")

# ============ Load dataset ============
df = pd.read_json(DATASET_FILE)

PROMPT_COL = "prompt_bn"
RESPONSE_COL = "chosen_response_bn"
if "prompt_safety" not in df.columns:
    raise ValueError("Expected 'prompt_safety' in dataset. Found: " + ", ".join(df.columns))

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["prompt_safety"]
)
train_df.to_json(TRAIN_SPLIT, orient="records", force_ascii=False, indent=2)
test_df.to_json(TEST_SPLIT, orient="records", force_ascii=False, indent=2)

print("Safe/Unsafe ratio (full):")
print(df["prompt_safety"].value_counts(normalize=True))

# ============ Model / Tokenizer (FULL FT) ============
MODEL_NAME = "hishab/titulm-llama-3.2-1b-v1.1"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # right padding for training

# Device / dtype
has_cuda = torch.cuda.is_available()
if has_cuda:
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)
    device_map = {"": LOCAL_RANK}
else:
    LOCAL_RANK = -1
    device_map = None

# Prefer bf16 if available (A100/H100), else fp16 on CUDA, else fp32 on CPU
bf16 = has_cuda and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16 else (torch.float16 if has_cuda else torch.float32)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,                   # ✅ use dtype (not torch_dtype)
    device_map=device_map,         # single GPU or None for CPU
    low_cpu_mem_usage=not has_cuda,
    trust_remote_code=True
)

# Full finetuning: everything trainable
model.train()
for p in model.parameters():
    p.requires_grad = True

# Memory savers
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# ⚠️ DO NOT torch.compile() — disables Inductor/Triton JIT to avoid your gcc/CUDA error
# If you later fix toolchain and want to try again, toggle this flag.
USE_TORCH_COMPILE = False
if USE_TORCH_COMPILE:
    try:
        model = torch.compile(model)  # may cause Triton/CUDA compile; leave off by default
    except Exception as e:
        print(f"[warn] torch.compile disabled due to: {e}")

# ============ Dataset & dynamic padding ============
dataset = load_dataset("json", data_files={"train": TRAIN_SPLIT, "test": TEST_SPLIT})

def format_examples(example):
    user = "[USER]: " + str(example[PROMPT_COL])
    assistant = "[ASSISTANT]: " + str(example[RESPONSE_COL])
    text = user + "\n" + assistant
    # No padding here; let the collator pad dynamically
    tok = tokenizer(text, truncation=True)
    return tok

tokenized = dataset.map(
    format_examples,
    batched=False,
    remove_columns=list(dataset["train"].column_names)
)

# Dynamic padding collator with pad-masked labels (-100)
def collate_fn(features, tokenizer=tokenizer, multiple_of=8):
    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=multiple_of  # helps Tensor Cores on NVIDIA
    )
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch

# ============ Training args (FULL FT) ============
# Use eval_strategy (transformers 4.56.x). If you upgrade, switch to evaluation_strategy.
training_args = TrainingArguments(
    output_dir=OUTPUT_ROOT,
    per_device_train_batch_size=2 if has_cuda else 1,   # tune to your VRAM
    per_device_eval_batch_size=2 if has_cuda else 1,
    gradient_accumulation_steps=8 if has_cuda else 32,  # increase if OOM
    learning_rate=2e-5,            # smaller LR for full FT
    num_train_epochs=3,
    bf16=bf16,
    fp16=(not bf16 and has_cuda),
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",          # ✅ 4.56.2 uses eval_strategy
    report_to="none",
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    remove_unused_columns=False,    # ✅ don't prune input_ids/attention_mask
    # deepspeed="ds_config.json",   # <- uncomment if you add a ZeRO config
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,     # future-proof vs tokenizer=
    data_collator=collate_fn
)

# ============ Baseline eval BEFORE training ============
baseline = trainer.evaluate()
print("📊 Baseline (pre-train):", baseline)
with open(os.path.join(OUTPUT_ROOT, "eval_before.json"), "w") as f:
    json.dump(baseline, f, indent=2)

# ============ Train ============
trainer.train()

# ============ Eval AFTER training ============
after = trainer.evaluate()
print("📊 After FT:", after)
with open(os.path.join(OUTPUT_ROOT, "eval_after.json"), "w") as f:
    json.dump(after, f, indent=2)

# ============ Simple generation helper (left-pad for decoder-only) ============
def generate_responses(prompts: List[str], max_new_tokens: int = 128, do_sample: bool = False):
    model.eval()
    outputs = []
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), 4), desc="Generating"):
            batch = prompts[i:i+4]
            inputs = tokenizer([f"[USER]: {p}\n[ASSISTANT]:" for p in batch],
                               return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model.generate(**inputs, **gen_kwargs)
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            for t in texts:
                parts = t.split("[ASSISTANT]:")
                outputs.append(parts[-1].strip() if len(parts) > 1 else t.strip())
    tokenizer.padding_side = orig_side
    return outputs

# ============ Save full model & tokenizer ============
trainer.save_model(OUTPUT_ROOT)     # saves full weights
tokenizer.save_pretrained(OUTPUT_ROOT)
print(f"✅ Full fine-tuned model saved to: {OUTPUT_ROOT}")

# (Optional) Quick sanity generation after training:
# prompts = ["এটা কি নিরাপদ?", "How to make a cake?"]
# print(generate_responses(prompts, max_new_tokens=64, do_sample=False))
