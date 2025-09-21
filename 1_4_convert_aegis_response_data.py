#!/usr/bin/env python3
"""
Convert the Aegis AI Content-Safety dataset responses to Bangla (robust, no truncation by default).

Key behavior:
  • Sanitizes inputs (URLs/emails/phones/markdown/control chars) to avoid tokenizer blowups
  • Enforces MAX_CHARS / MAX_TOKENS on the *source*
      - Default: if exceeded -> SKIP (no truncation)
      - Optional: CHUNK_OVER_LIMIT=True to split long texts into token-safe windows and translate piecewise
  • Translates with safer generation params (max_new_tokens, no sampling)
  • On ANY translation error -> SKIP the record AND rebuild the translator so CUDA state resets
  • Logs all skipped records (with reason) and prints reason breakdown as it goes
  • Saves kept translations to datasets/converted_aegis_response_bangla.json
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset  # type: ignore
from transformers import pipeline, AutoTokenizer  # type: ignore

# ---------------- CONFIG (tune here) ----------------
MODEL_NAME      = os.getenv("AegisTranslatorModel", "shhossain/opus-mt-en-to-bn")
GPU_INDEX       = int(os.getenv("AegisGPUIndex", "0"))
DEVICE          = 0 if torch.cuda.is_available() else -1  # 0=CUDA, -1=CPU

# Source-side limits (before translation)
MAX_CHARS       = 2000     # char cap; if exceeded => skip (unless chunking enabled)
MAX_TOKENS      = 512     # token cap; if exceeded => skip (unless chunking enabled)

# Behavior toggles
SANITIZE        = True
SKIP_ON_ERROR   = True     # drop any translation error (recommended)
CHUNK_OVER_LIMIT= False    # False=skip over-limit; True=split into token-sized chunks and translate piecewise

# Chunking parameters (only used if CHUNK_OVER_LIMIT=True)
CHUNK_TOKENS    = 512      # size of each chunk in tokens
CHUNK_OVERLAP   = 32       # token overlap between chunks (to avoid split artifacts)

# Generation parameters
MAX_NEW_TOKENS  = 256
NUM_BEAMS       = 3

OUT_PATH        = Path("datasets/converted_aegis_response_bangla.json")
SKIPPED_LOG     = Path("datasets/translation_skipped_ids.txt")
PROGRESS_EVERY  = 200
# ----------------------------------------------------

# ---- sanitization helpers (non-destructive normalization) ----
_URL_RE    = re.compile(r'https?://\S+')
_EMAIL_RE  = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_PHONE_RE  = re.compile(r'\+?\d[\d\-\s\(\)]{7,}\d')
_CONTROL   = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

def sanitize_text(s: str) -> str:
    s = _URL_RE.sub('<URL>', s)
    s = _EMAIL_RE.sub('<EMAIL>', s)
    s = _PHONE_RE.sub('<PHONE>', s)
    s = s.replace('\u200b', '')                  # zero-width
    s = s.replace('**', '').replace('###', ' ').replace('`', '')
    s = _CONTROL.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s
# ---------------------------------------------------------------

def build_translator(model_name: str, tok) -> Any:
    if DEVICE == 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_INDEX))
    return pipeline("translation", model=model_name, tokenizer=tok, device=DEVICE)

def translate_one(text: str, translator) -> str:
    out = translator(
        text,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=False,
        clean_up_tokenization_spaces=True,
    )[0]
    return out["translation_text"]

def translate_many(texts: List[str], translator) -> List[str]:
    outs = translator(
        texts,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=False,
        clean_up_tokenization_spaces=True,
    )
    return [o["translation_text"] for o in outs]

def tokens(s: str, tok: AutoTokenizer) -> List[int]:
    return tok.encode(s, add_special_tokens=False)

def detok(ids: List[int], tok: AutoTokenizer) -> str:
    return tok.decode(ids, skip_special_tokens=True)

def chunk_by_tokens(s: str, tok: AutoTokenizer, chunk_tokens: int, overlap: int) -> List[str]:
    """Split text into token windows <= chunk_tokens (with overlap), then detokenize each window back to text."""
    ids = tokens(s, tok)
    if len(ids) <= chunk_tokens:
        return [s]
    chunks = []
    i = 0
    step = max(1, chunk_tokens - overlap)
    while i < len(ids):
        window = ids[i: i + chunk_tokens]
        chunks.append(detok(window, tok))
        i += step
    return chunks

def parse_categories(violated: Optional[str]) -> Tuple[str, Optional[str]]:
    if not violated:
        return "Safe", None
    parts = [p.strip() for p in str(violated).split(",") if p.strip()]
    if not parts:
        return "Safe", None
    return parts[0], (parts[1] if len(parts) > 1 else None)

def main() -> None:
    print("Loading Aegis training split…")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")

    print(f"Initialising tokenizer/model: {MODEL_NAME} (device={'CUDA' if DEVICE==0 else 'CPU'})")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    translator = build_translator(MODEL_NAME, tok)

    records: List[Dict[str, Any]] = [ex for ex in ds]
    converted: List[Dict[str, Optional[str]]] = []
    skipped_lines: List[str] = []
    skip_counts = {
        "missing_fields": 0,
        "over_char_limit": 0,
        "over_token_limit": 0,
        "translate_error": 0,
    }

    total = len(records)
    print(f"Processing {total} records…")

    for idx, ex in enumerate(records):
        rid = idx + 1

        # only keep entries with response + response_label
        resp = ex.get("response")
        lab  = ex.get("response_label")
        if not resp or not lab:
            skip_counts["missing_fields"] += 1
            skipped_lines.append(f"{rid}\tmissing_fields")
            continue

        text = str(resp)
        if SANITIZE:
            text = sanitize_text(text)

        # char cap -> skip or chunk
        if MAX_CHARS is not None and len(text) > MAX_CHARS:
            if not CHUNK_OVER_LIMIT:
                skip_counts["over_char_limit"] += 1
                skipped_lines.append(f"{rid}\tover_char_limit({len(text)}>{MAX_CHARS})")
                continue
            # If chunking, we still proceed (token checks below)

        # token cap -> skip or chunk
        src_ids = tokens(text, tok) if MAX_TOKENS is not None else []
        if MAX_TOKENS is not None and len(src_ids) > MAX_TOKENS:
            if not CHUNK_OVER_LIMIT:
                skip_counts["over_token_limit"] += 1
                skipped_lines.append(f"{rid}\tover_token_limit({len(src_ids)}>{MAX_TOKENS})")
                continue

        try:
            if CHUNK_OVER_LIMIT and MAX_TOKENS is not None and len(src_ids) > MAX_TOKENS:
                # Piecewise translate, then join
                parts = chunk_by_tokens(text, tok, CHUNK_TOKENS, CHUNK_OVERLAP)
                # You can batch here, but we keep it simple and robust
                translated_parts = []
                for p in parts:
                    translated_parts.append(translate_one(p, translator))
                bn = " ".join(translated_parts)
            else:
                bn = translate_one(text, translator)
        except Exception as e:
            skip_counts["translate_error"] += 1
            skipped_lines.append(f"{rid}\ttranslate_error({type(e).__name__})")
            # reset translator so next items don't inherit a poisoned CUDA state
            try:
                del translator
                if DEVICE == 0:
                    torch.cuda.empty_cache()
            finally:
                translator = build_translator(MODEL_NAME, tok)
            continue

        primary, secondary = parse_categories(ex.get("violated_categories"))
        converted.append({
            "id": str(rid),
            "response": bn,
            "label": str(lab).lower(),
            "category": primary,
            "sub_category": secondary,
        })

        if rid % PROGRESS_EVERY == 0:
            kept = len(converted)
            skipped = sum(skip_counts.values())
            print(f"  Processed {rid}/{total}… kept={kept} skipped={skipped} "
                  f"[missing={skip_counts['missing_fields']}, "
                  f"char={skip_counts['over_char_limit']}, "
                  f"tok={skip_counts['over_token_limit']}, "
                  f"err={skip_counts['translate_error']}]")

    # write outputs
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"responses": converted}, f, ensure_ascii=False, indent=2)

    if skipped_lines:
        SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SKIPPED_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_lines))

    print(f"✅ Kept {len(converted)} responses → {OUT_PATH}")
    print(f"⚠️ Skipped {sum(skip_counts.values())} responses "
          f"(missing={skip_counts['missing_fields']}, "
          f"char={skip_counts['over_char_limit']}, "
          f"tok={skip_counts['over_token_limit']}, "
          f"err={skip_counts['translate_error']}) → {SKIPPED_LOG}")

if __name__ == "__main__":
    main()
