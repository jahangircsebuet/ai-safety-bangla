#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast, zero-argument converter for Aegis-AI Content Safety (train split) -> Bangla prompts.

Run:
    python 1_4_convert_ageis_dataset

Output:
    ./datasets/converted_aegis_bangla.json
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# keep tokenizers quiet in multi-proc
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- defaults (tweak in-code if you like) ----
OUTPUT_PATH: str = "./datasets/converted_aegis_bangla.json"
MODEL_ID: str = "Helsinki-NLP/opus-mt-en-bn"
MAX_EXAMPLES: Optional[int] = None   # e.g., 2000 for a quick test; None = full train
BATCH_SIZE: int = 96                 # raise if you have more VRAM (96/128)
MAX_LENGTH: int = 768                # kept for reference, no longer used in gen_kwargs
OUT_MAX_TOKENS: int = 512            # 🚀 generation budget per item (prevents the warning)
INPUT_MAX_TOKENS: int = 1024         # pre-truncate inputs to avoid OOM (keep if you want)
NUM_BEAMS: int = 1                   # keep 1 for speed
CHECKPOINT_EVERY: int = 4000         # write partial file every N items; 0 disables
DEVICE: Optional[int] = None         # 0=cuda:0, -1=cpu, None=auto (respects CUDA_VISIBLE_DEVICES)
FP16_DEFAULT: bool = True            # use fp16 on GPU by default

# ---- deps ----
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

try:
    import torch
except Exception:
    torch = None


def _load_token() -> str:
    """Load HF token from env or .env in current/parent dirs."""
    tok = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    for candidate in (".env", "../.env", "../../.env"):
        p = Path(candidate)
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.startswith("HUGGING_FACE_HUB_TOKEN="):
                        return line.split("=", 1)[1].strip()
            except Exception:
                pass
    return ""


def _resolve_device(requested: Optional[int]) -> int:
    """
    Map requested device to an actual pipeline device index, respecting CUDA_VISIBLE_DEVICES.
    - If CUDA_VISIBLE_DEVICES is set (e.g., "2"), the first visible GPU is always index 0.
    - requested == -1 forces CPU.
    - requested is ignored when CUDA_VISIBLE_DEVICES is set (except -1): we use 0.
    """
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if requested == -1:
        return -1
    if visible != "":
        # Mask is active; cuda:0 refers to the first visible GPU in the mask.
        return 0
    # No mask; use requested if provided, else auto (GPU if available else CPU)
    if requested is not None:
        return requested
    if torch is not None and torch.cuda.is_available():
        return 0
    return -1


def _init_translator(model_id: str, device: Optional[int], fp16_default: bool):
    """Build EN->BN translation pipeline with sensible defaults & fallbacks."""
    token = _load_token()
    resolved_device = _resolve_device(device)

    # fp16 if GPU available and torch present
    model_kwargs: Dict[str, Any] = {}
    if fp16_default and resolved_device == 0 and (torch is not None) and torch.cuda.is_available():
        try:
            model_kwargs["torch_dtype"] = torch.float16
        except Exception:
            pass

    candidates = [
        model_id,
        "Helsinki-NLP/opus-mt-en-bengali",
        "facebook/nllb-200-distilled-600M",  # fallback multilingual
    ]

    last_err = None
    for mid in candidates:
        try:
            if "nllb" in mid:
                return (
                    pipeline(
                        "translation",
                        model=mid,
                        token=token or None,
                        src_lang="eng_Latn",
                        tgt_lang="ben_Beng",
                        device=resolved_device,
                        model_kwargs=model_kwargs,
                    ),
                    resolved_device,
                )
            return (
                pipeline(
                    "translation",
                    model=mid,
                    token=token or None,
                    device=resolved_device,
                    model_kwargs=model_kwargs,
                ),
                resolved_device,
            )
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load any translation model. Last error: {last_err}")


def _normalize_label(lab: Any) -> str:
    return "safe" if str(lab).strip().lower() == "safe" else "unsafe"


def _truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """
    Truncate `text` to <= max_tokens using the model tokenizer (wordpiece-aware),
    then decode back to string so the pipeline ingests safe-length input.
    """
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    print("Loading Aegis-AI-Content-Safety-Dataset-2.0 (train split)...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    print(f"Loaded {len(ds)} examples.")

    # filter once so streaming is clean
    def _ok(ex):
        return (ex.get("prompt") is not None) and (ex.get("prompt_label") is not None)

    ds = ds.filter(_ok)
    total = len(ds)
    if MAX_EXAMPLES is not None:
        total = min(total, MAX_EXAMPLES)
        ds = ds.select(range(total))
    print(f"Total to translate: {total}")

    print("Initialising translation model…")
    translator, resolved_device = _init_translator(MODEL_ID, DEVICE, FPAL1_DEFAULT if 'FPAL1_DEFAULT' in globals() else FP16_DEFAULT)
    # ^ small guard in case you paste over an env where a typo exists—falls back to FP16_DEFAULT
    dev_str = "cuda:0" if resolved_device == 0 else "cpu"
    print(f"Translation model ready on {dev_str}")

    # Pre-truncate the inputs to a safe token length to avoid OOM
    tok = translator.tokenizer
    model_cap = getattr(tok, "model_max_length", INPUT_MAX_TOKENS)
    input_cap = min(INPUT_MAX_TOKENS, model_cap if model_cap and model_cap < 10**9 else INPUT_MAX_TOKENS)

    def _trim_batch(batch):
        prompts = batch["prompt"]
        trimmed = [_truncate_to_tokens(p, tok, input_cap) for p in prompts]
        batch["prompt_trim"] = trimmed
        return batch

    ds = ds.map(_trim_batch, batched=True, remove_columns=[])
    print(f"Inputs will be truncated to ≤ {input_cap} tokens per item.")

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []
    processed = 0
    cursor = 0  # index into ds

    # ✅ Quickest fix: use max_new_tokens (not max_length) to avoid the warning
    gen_kwargs = dict(max_new_tokens=OUT_MAX_TOKENS, num_beams=NUM_BEAMS)

    # Stream through dataset, batching handled by pipeline with KeyDataset
    stream = translator(KeyDataset(ds, "prompt_trim"), batch_size=BATCH_SIZE, **gen_kwargs)

    for out in stream:
        # pipeline yields dict per item in stream mode
        bn = out["translation_text"] if isinstance(out, dict) else str(out)
        lab = ds[cursor]["prompt_label"]
        results.append({"prompt": bn, "label": _normalize_label(lab), "source": "aegis"})
        cursor += 1
        processed += 1

        if processed % 200 == 0:
            print(f"Processed {processed}/{total}…")

        if CHECKPOINT_EVERY and (processed % CHECKPOINT_EVERY == 0):
            partial = OUTPUT_PATH + ".partial.json"
            with open(partial, "w", encoding="utf-8") as f:
                json.dump({"prompts": results}, f, ensure_ascii=False, indent=2)
            print(f"[checkpoint] wrote {partial} at {processed} examples")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"prompts": results}, f, ensure_ascii=False, indent=2)
    print(f"✅ Finished conversion. Saved {len(results)} items to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
