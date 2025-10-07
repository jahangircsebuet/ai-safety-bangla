#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TigerLLM-1B-it Safety Fine-Tuning (LoRA + CE + Contrastive) — Gemma3 Causal LM
---------------------------------------------------------------------------------
This script fine-tunes TigerLLM-1B-it (Gemma3 causal LM) using a combination of
cross-entropy and margin-based contrastive loss.
"""

import os, json, math, random, warnings
from typing import List, Dict
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# ===============================================================
# Config
# ===============================================================
BASE_MODEL_NAME = "md-nishat-008/TigerLLM-1B-it"
RESULTS_BASE_DIR = "./FineTuned_Models"
LOSS_TYPE = "_LoRA_CE+Contrastive"
SAVE_FOLDER_NAME = BASE_MODEL_NAME.split("/")[-1] + LOSS_TYPE

EPOCHS = 5
BATCH_SIZE = 16
GRAD_ACCUM = 8
LR = 5e-5
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = not USE_BF16
MAX_SOURCE_LEN = 2048
MAX_TARGET_LEN = 512
MAX_TOTAL_LEN = MAX_SOURCE_LEN + MAX_TARGET_LEN
GAMMA = 5.0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFUSAL_TEMPLATE = (
    "এই প্রশ্নটি বিপজ্জনক। এর উত্তরে অসুরক্ষিত বিষয় হতে পারে। "
    "দয়া করে নিরাপদ প্রথা অনুসরণ করুন।"
)
warnings.filterwarnings("ignore", category=FutureWarning)
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ===============================================================
# AMP setup
# ===============================================================
try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_ARGS = {"device_type": "cuda", "dtype": torch.bfloat16 if USE_BF16 else torch.float16}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AUTOCAST_ARGS = {"dtype": torch.float16}
scaler = GradScaler(enabled=torch.cuda.is_available())

# ===============================================================
# Load Model + Tokenizer
# ===============================================================
print(f"Loading {BASE_MODEL_NAME} (Gemma3 Causal LM) ...")
config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
config.use_cache = False; config.attn_implementation = "sdpa"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id or 0
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, config=config,
    torch_dtype=(torch.bfloat16 if USE_BF16 else None),
)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config).to(DEVICE)
model.train()

# ===============================================================
# Dataset loader helpers
# ===============================================================
def load_dataset_from_list(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        input_texts = [d.get("prompt") or d.get("input_text", "") for d in data]
        output_texts = [d.get("response") or d.get("output_text", "") for d in data]
        p_labels = [d.get("prompt_label", "safe") for d in data]
        p_cats = [d.get("prompt_category", "safe") for d in data]
        return Dataset.from_dict({
            "input_text": input_texts, "output_text": output_texts,
            "prompt_label": p_labels, "prompt_category": p_cats,
        })
    elif isinstance(data, dict):
        return Dataset.from_dict(data)
    else:
        raise ValueError(f"Unsupported JSON format: {path}")

# ===============================================================
# Load Datasets
# ===============================================================
train_paths = [
    "/home/tahad/ai-safety-bangla/Final_dataset/train.json",
    "/home/tahad/ai-safety-bangla/Final_dataset/Unsafe_responses.json",
]
VAL_DATA_PATH = "/home/tahad/ai-safety-bangla/Final_dataset/val.json"
TEST_DATA_PATH = "/home/tahad/ai-safety-bangla/Final_dataset/test.json"

raw_train_items = []
for path in train_paths:
    with open(path, "r", encoding="utf-8") as f:
        raw_train_items.extend(json.load(f))
if not raw_train_items:
    raise ValueError("No training data found!")

random.shuffle(raw_train_items)
def encode_entry(e: Dict): 
    if e.get("prompt_label") == "unsafe":
        if e.get("response_label") == "safe": resp = e["response"]
        else: resp = REFUSAL_TEMPLATE if random.random() < 0.5 else e["response"]
    else: resp = e["response"]
    return {"input_text": e["prompt"], "output_text": resp, "prompt_label": e.get("prompt_label", "safe")}
encoded_train = [encode_entry(e) for e in raw_train_items]
train_dataset = Dataset.from_dict({
    "input_text": [e["input_text"] for e in encoded_train],
    "output_text": [e["output_text"] for e in encoded_train],
    "prompt_label": [e["prompt_label"] for e in encoded_train],
})
val_dataset = load_dataset_from_list(VAL_DATA_PATH)
test_dataset = load_dataset_from_list(TEST_DATA_PATH)

save_root = os.path.join(RESULTS_BASE_DIR, SAVE_FOLDER_NAME)
model_dir = os.path.join(save_root, "model")
results_dir = os.path.join(save_root, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "val_dataset.json"), "w", encoding="utf-8") as f: json.dump(val_dataset.to_dict(), f, indent=2, ensure_ascii=False)
with open(os.path.join(results_dir, "test_dataset.json"), "w", encoding="utf-8") as f: json.dump(test_dataset.to_dict(), f, indent=2, ensure_ascii=False)
print(f"✅ Loaded Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ===============================================================
# Build Causal Batch + Log-Likelihood
# ===============================================================
def build_causal_batch(prompts, targets, tokenizer, max_src, max_tgt, device):
    enc_p = tokenizer(prompts, padding=True, truncation=True, max_length=max_src, return_tensors="pt")
    enc_t = tokenizer(targets, padding=True, truncation=True, max_length=max_tgt, add_special_tokens=False, return_tensors="pt")
    pad_id = tokenizer.pad_token_id
    input_ids, attn, labels = [], [], []
    for p, t in zip(enc_p.input_ids, enc_t.input_ids):
        if tokenizer.eos_token_id and (len(t)==0 or t[-1]!=tokenizer.eos_token_id):
            t = torch.cat([t, torch.tensor([tokenizer.eos_token_id])])
        combined = torch.cat([p, t])[:MAX_TOTAL_LEN]
        mask = torch.ones_like(combined)
        lab = combined.clone(); lab[:len(p)] = -100
        input_ids.append(combined); attn.append(mask); labels.append(lab)
    max_len = max(len(x) for x in input_ids)
    def pad(seq, val): out = torch.full((len(seq), max_len), val, dtype=torch.long); [out[i,:len(s)].copy_(s) for i,s in enumerate(seq)]; return out
    return {"input_ids": pad(input_ids, pad_id).to(device),
            "attention_mask": pad(attn, 0).to(device),
            "labels": pad(labels, -100).to(device)}

@torch.no_grad()
def batch_loglikelihood(prompts, targets, model, tokenizer, device):
    model.eval(); scores=[]
    for i in range(0, len(prompts), 4):
        bp, bt = prompts[i:i+4], targets[i:i+4]
        b = build_causal_batch(bp, bt, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, device)
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits
        logp = F.log_softmax(logits[:,:-1,:], dim=-1)
        labels = b["labels"][:,1:]; valid = labels!=-100
        labels_ = labels.clone(); labels_[~valid]=0
        tokp = torch.gather(logp, -1, labels_.unsqueeze(-1)).squeeze(-1)
        tokp = tokp*valid.float(); scores.append(tokp.sum(1).cpu())
    model.train(); return torch.cat(scores,0).to(device)

# ===============================================================
# Training
# ===============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
steps_per_epoch = math.ceil(len(train_dataset)/(BATCH_SIZE*GRAD_ACCUM))
ce_hist, con_hist, tot_hist = [],[],[]
print(f"=== Training {EPOCHS} epochs ({steps_per_epoch} steps/epoch) ===")

for epoch in range(EPOCHS):
    ce_e, con_e = [], []
    for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}"):
        batch = train_dataset.shuffle(seed=epoch*100+step).select(range(BATCH_SIZE)).to_dict()
        p, t, lbl = batch["input_text"], batch["output_text"], batch["prompt_label"]
        b = build_causal_batch(p, t, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, DEVICE)
        with autocast(**AUTOCAST_ARGS, enabled=torch.cuda.is_available()):
            out = model(**b); ce = out.loss
        unsafe = [i for i,x in enumerate(lbl) if x=="unsafe"]
        con_loss = torch.tensor(0.0, device=DEVICE)
        if unsafe:
            pu=[p[i] for i in unsafe]; pos=[REFUSAL_TEMPLATE]*len(unsafe); neg=[t[i] for i in unsafe]
            s_pos=batch_loglikelihood(pu,pos,model,tokenizer,DEVICE); s_neg=batch_loglikelihood(pu,neg,model,tokenizer,DEVICE)
            con_loss=torch.clamp(GAMMA-(s_pos-s_neg),min=0).mean()
        tot = ce+con_loss
        scaler.scale(tot/GRAD_ACCUM).backward()
        if (step+1)%GRAD_ACCUM==0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        ce_e.append(ce.item()); con_e.append(float(con_loss))
    ce_hist.append(sum(ce_e)/len(ce_e)); con_hist.append(sum(con_e)/len(con_e))
    tot_hist.append(ce_hist[-1]+con_hist[-1])
    print(f"Epoch {epoch+1}: CE={ce_hist[-1]:.4f}, Contrastive={con_hist[-1]:.4f}")
    model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)

# ===============================================================
# Plot losses
# ===============================================================
plt.figure(figsize=(6,4))
plt.plot(ce_hist,label="CE"); plt.plot(con_hist,label="Contrastive"); plt.plot(tot_hist,label="Total")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True,alpha=0.3)
plt.title("Training Loss Components (LoRA Gemma3)")
plt.tight_layout(); plt.savefig(os.path.join(results_dir,"training_loss_curve.png"),dpi=300)
print("✅ Training complete.")
