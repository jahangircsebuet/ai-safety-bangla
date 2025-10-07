#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TigerLLM-1B-it Safety Fine-Tuning (LoRA + CE + Contrastive)
------------------------------------------------------------
Trains the model for several predefined GAMMA values (margin strengths) and
records CE and Contrastive losses per epoch for comparative analysis.
Each run starts from a fresh LoRA initialization to ensure fair evaluation.
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
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluate these margin strengths
GAMMA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0]

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
# Load Dataset
# ===============================================================
def load_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        input_texts = [d.get("prompt") or d.get("input_text", "") for d in data]
        output_texts = [d.get("response") or d.get("output_text", "") for d in data]
        p_labels = [d.get("prompt_label", "safe") for d in data]
        return Dataset.from_dict({
            "input_text": input_texts,
            "output_text": output_texts,
            "prompt_label": p_labels,
        })
    elif isinstance(data, dict):
        return Dataset.from_dict(data)
    else:
        raise ValueError("Invalid JSON format.")

train_paths = [
    "/home/tahad/ai-safety-bangla/Final_dataset/train.json",
    "/home/tahad/ai-safety-bangla/Final_dataset/Unsafe_responses.json",
]
train_data = []
for p in train_paths:
    with open(p, "r", encoding="utf-8") as f:
        train_data.extend(json.load(f))
random.shuffle(train_data)

def encode_entry(e):
    if e.get("prompt_label") == "unsafe":
        if e.get("response_label") == "safe":
            resp = e["response"]
        else:
            resp = REFUSAL_TEMPLATE if random.random() < 0.5 else e["response"]
    else:
        resp = e["response"]
    return {"input_text": e["prompt"], "output_text": resp, "prompt_label": e.get("prompt_label", "safe")}

encoded = [encode_entry(e) for e in train_data]
train_dataset = Dataset.from_dict({
    "input_text": [e["input_text"] for e in encoded],
    "output_text": [e["output_text"] for e in encoded],
    "prompt_label": [e["prompt_label"] for e in encoded],
})
print(f"✅ Loaded training dataset with {len(train_dataset)} samples.")

# ===============================================================
# Utility: build causal batch + log-likelihood
# ===============================================================
def build_causal_batch(prompts, targets, tokenizer):
    enc_p = tokenizer(prompts, padding=True, truncation=True,
                      max_length=MAX_SOURCE_LEN, return_tensors="pt")
    enc_t = tokenizer(targets, padding=True, truncation=True,
                      max_length=MAX_TARGET_LEN, add_special_tokens=False, return_tensors="pt")
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
    def pad(seq, val):
        out = torch.full((len(seq), max_len), val, dtype=torch.long)
        for i,s in enumerate(seq): out[i,:len(s)] = s
        return out
    return {"input_ids": pad(input_ids, pad_id).to(DEVICE),
            "attention_mask": pad(attn, 0).to(DEVICE),
            "labels": pad(labels, -100).to(DEVICE)}

@torch.no_grad()
def batch_loglikelihood(prompts, targets, model, tokenizer):
    model.eval(); scores=[]
    for i in range(0, len(prompts), 4):
        bp, bt = prompts[i:i+4], targets[i:i+4]
        b = build_causal_batch(bp, bt, tokenizer)
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits
        logp = F.log_softmax(logits[:,:-1,:], dim=-1)
        labels = b["labels"][:,1:]
        valid = (labels!=-100)
        labels_ = labels.clone(); labels_[~valid]=0
        tokp = torch.gather(logp, -1, labels_.unsqueeze(-1)).squeeze(-1)
        tokp = tokp*valid.float(); scores.append(tokp.sum(1).cpu())
    model.train(); return torch.cat(scores,0).to(DEVICE)

# ===============================================================
# Multi-GAMMA Training Loop
# ===============================================================
save_root = os.path.join(RESULTS_BASE_DIR, SAVE_FOLDER_NAME)
os.makedirs(save_root, exist_ok=True)
results_summary = []

for GAMMA in GAMMA_VALUES:
    print(f"\n Starting fine-tuning with GAMMA = {GAMMA}")

    # ---- fresh LoRA model each run ----
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
    config.use_cache = False; config.attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, config=config,
        torch_dtype=(torch.bfloat16 if USE_BF16 else None),
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    ce_hist, con_hist = [], []
    steps_per_epoch = math.ceil(len(train_dataset)/(BATCH_SIZE*GRAD_ACCUM))

    for epoch in range(EPOCHS):
        ce_e, con_e = [], []
        for step in tqdm(range(steps_per_epoch), desc=f"γ={GAMMA} | Epoch {epoch+1}"):
            batch = train_dataset.shuffle(seed=epoch*100+step).select(range(BATCH_SIZE)).to_dict()
            p, t, lbl = batch["input_text"], batch["output_text"], batch["prompt_label"]
            b = build_causal_batch(p, t, tokenizer)
            with autocast(**AUTOCAST_ARGS, enabled=torch.cuda.is_available()):
                out = model(**b); ce = out.loss
            unsafe_idx = [i for i,x in enumerate(lbl) if x=="unsafe"]
            con_loss = torch.tensor(0.0, device=DEVICE)
            if unsafe_idx:
                pu=[p[i] for i in unsafe_idx]
                pos=[REFUSAL_TEMPLATE]*len(unsafe_idx)
                neg=[t[i] for i in unsafe_idx]
                s_pos=batch_loglikelihood(pu,pos,model,tokenizer)
                s_neg=batch_loglikelihood(pu,neg,model,tokenizer)
                con_loss=torch.clamp(GAMMA-(s_pos-s_neg),min=0).mean()
            total=ce+con_loss
            scaler.scale(total/GRAD_ACCUM).backward()
            if (step+1)%GRAD_ACCUM==0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            ce_e.append(ce.item()); con_e.append(float(con_loss))
        ce_mean=sum(ce_e)/len(ce_e); con_mean=sum(con_e)/len(con_e)
        ce_hist.append(ce_mean); con_hist.append(con_mean)
        results_summary.append({"gamma":GAMMA,"epoch":epoch+1,
                                "ce_loss":ce_mean,"contrastive_loss":con_mean})
        print(f"Epoch {epoch+1} (γ={GAMMA}): CE={ce_mean:.4f}, Contrastive={con_mean:.4f}")

    # save checkpoint for each gamma
    gamma_dir = os.path.join(save_root, f"gamma_{GAMMA}")
    os.makedirs(gamma_dir, exist_ok=True)
    model.save_pretrained(gamma_dir); tokenizer.save_pretrained(gamma_dir)

# ===============================================================
# Save & Plot Results
# ===============================================================
summary_path = os.path.join(save_root,"gamma_loss_summary.json")
with open(summary_path,"w",encoding="utf-8") as f:
    json.dump(results_summary,f,indent=2,ensure_ascii=False)
print(f"\n Saved epoch-wise CE and Contrastive losses for all GAMMA runs → {summary_path}")

plt.figure(figsize=(7,5))
for g in GAMMA_VALUES:
    ce_points=[r["ce_loss"] for r in results_summary if r["gamma"]==g]
    con_points=[r["contrastive_loss"] for r in results_summary if r["gamma"]==g]
    plt.plot(ce_points,label=f"CE γ={g}",linestyle="--")
    plt.plot(con_points,label=f"Con γ={g}")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("CE vs Contrastive Loss across GAMMA values")
plt.legend(); plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_root,"gamma_loss_curve.png"),dpi=300)
print(" All GAMMA fine-tuning runs complete.")



#CUDA_VISIBLE_DEVICES=2 nohup python 1_3_7_1_Gamma_value_finder.py > /home/tahad/ai-safety-bangla/logs/1_3_7_1_Gamma_value_finder.log 2>&1 &