#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert the Aegis AI Content Safety (train split) prompts to Bangla using a
translation model. Only rows with BOTH prompt_label and response_label are kept.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any

from datasets import load_dataset  # type: ignore
from transformers import pipeline  # type: ignore


def translate_batch(texts: List[str], translator) -> List[str]:
    outputs = translator(texts, max_length=512)
    return [o["translation_text"] for o in outputs]


def parse_categories(violated: Optional[str]) -> (str, Optional[str]):
    if not violated:
        return "Safe", None
    parts = [p.strip() for p in violated.split(",") if p.strip()]
    if not parts:
        return "Safe", None
    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else None
    return primary, secondary


def main() -> None:
    print("Loading Aegis training split…")
    dataset = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train"
    )

    import os
    model_name = os.getenv("AegisTranslatorModel", "shhossain/opus-mt-en-to-bn")
    print(f"Initialising translation model: {model_name}")
    translator = pipeline("translation", model=model_name, device=0)

    records: List[Dict[str, Any]] = [ex for ex in dataset]
    converted: List[Dict[str, Optional[str]]] = []
    total = len(records)
    print(f"Processing {total} prompts…")

    for idx, ex in enumerate(records):
        # ✅ filter condition: skip if prompt_label or response_label is missing
        if not ex.get("prompt_label") or not ex.get("response_label"):
            continue

        eng = ex.get("prompt", "") or ""
        label = ex.get("prompt_label", "unsafe") or "unsafe"
        vio = ex.get("violated_categories")

        if len(eng) > 1000:
            eng_trunc = eng[:1000]
        else:
            eng_trunc = eng
        try:
            bn = translator(eng_trunc, max_length=512)[0]["translation_text"]
        except Exception as e:
            print(f"  ⚠️ Error translating record {idx + 1}: {e}. Using truncated English text.")
            bn = eng_trunc

        primary, secondary = parse_categories(vio)
        entry = {
            "id": str(idx + 1),
            "prompt": bn,
            "label": label,
            "category": primary,
            "sub_category": secondary,
        }
        converted.append(entry)

        if (idx + 1) % 100 == 0:
            print(f"  Translated {idx + 1} / {total} prompts…")

    output = {"prompts": converted}
    output_path = Path("datasets/converted_aegis_bangla.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ Converted {len(converted)} prompts (kept only rows with both labels) and saved to {output_path}")


if __name__ == "__main__":
    main()
