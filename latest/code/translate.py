import json
import math
import torch
import re
from typing import Dict, Any, Optional
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# MODEL PRESETS (extend as needed)
# -----------------------------
MODEL_PRESETS = {
    # M2M100
    "m2m100_418m": {
        "model_name": "facebook/m2m100_418M",
        "src_lang": "bn",
        "tgt_lang": "sw",
        "lang_key": "sw",
    },

    # NLLB-200 family
    "nllb200_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
    "nllb200_1.3b": {
        "model_name": "facebook/nllb-200-1.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
    "nllb200_3.3b": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
}


MODEL_PRESETS.update({
    "nllb200_3.3b_bn_sw": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
    "nllb200_3.3b_bn_ha": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "hau_Latn",
        "lang_key": "ha",
    },
    "nllb200_3.3b_bn_en": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "ben_Beng",
        "tgt_lang": "eng_Latn",
        "lang_key": "en",
    },
    "nllb200_3.3b_en_sw": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "swh_Latn",
        "lang_key": "sw",
    },
    "nllb200_3.3b_en_ha": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "hau_Latn",
        "lang_key": "ha",
    },
    "nllb200_3.3b_en_bn": {
        "model_name": "facebook/nllb-200-3.3B",
        "src_lang": "eng_Latn",
        "tgt_lang": "ben_Beng",
        "lang_key": "bn",
    },
})

# -----------------------------
# Sentence splitting (Bangla-friendly)
# -----------------------------
_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+|\n+")


def _split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in _SPLIT_RE.split(text.strip()) if s.strip()]


def _build_translator(model_name: str, src_lang: str, tgt_lang: str, device: int = 0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ✅ half precision on GPU
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float16,   # dtype, not torch_dtype
    ).to(f"cuda:{device}")
    model.eval()


    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,         # OK now
    )

    return translator


split_prompt_count, split_response_count = 0, 0

def split_sentences_bn(text: str, is_prompt: bool = True):
    """Split text into sentences and update counters."""
    global split_prompt_count, split_response_count

    if not isinstance(text, str) or not text.strip():
        return []

    sentences = [s.strip() for s in _SPLIT_RE.split(text.strip()) if s.strip()]

    if len(sentences) > 1:
        if is_prompt:
            split_prompt_count += 1
        else:
            split_response_count += 1

    return sentences

def _nllb_forced_bos_id(tokenizer, tgt_lang: str) -> int:
    """
    Robustly map NLLB language code (e.g., 'hau_Latn') to the BOS token id
    across tokenizer variants / versions.
    """
    # 1) try vocab lookup
    vocab = tokenizer.get_vocab()
    if tgt_lang in vocab:
        return vocab[tgt_lang]

    # 2) try token->id conversion
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if bos_id is not None and bos_id != tokenizer.unk_token_id:
        return bos_id

    # 3) common alt: some tokenizers store as special token like "__hau_Latn__" (rare)
    alt = f"__{tgt_lang}__"
    if alt in vocab:
        return vocab[alt]
    bos_id = tokenizer.convert_tokens_to_ids(alt)
    if bos_id is not None and bos_id != tokenizer.unk_token_id:
        return bos_id

    raise KeyError(
        f"Could not find BOS id for tgt_lang='{tgt_lang}'. "
        f"Check you are using an NLLB tokenizer and the code is valid (e.g., eng_Latn, hau_Latn)."
    )


def batch_translate_texts_patched(
    translator,
    texts,
    *,
    src_lang: str,
    tgt_lang: str,
    is_prompt=True,
    sent_batch_size=32,
    max_new_tokens=256,
    truncation_max_length=256,
):
    batched_sentences, indices = [], []
    for i, text in enumerate(texts):
        for sent in split_sentences_bn(text, is_prompt=is_prompt):
            sent = sent.strip()
            if sent:
                batched_sentences.append(sent)
                indices.append(i)

    if not batched_sentences:
        return texts

    tokzr = translator.tokenizer
    model = translator.model
    device = model.device

    # ✅ NLLB needs src_lang and forced_bos_token_id
    if hasattr(tokzr, "src_lang"):
        tokzr.src_lang = src_lang

    forced_bos = _nllb_forced_bos_id(tokzr, tgt_lang)

    outputs = [""] * len(texts)

    for start in range(0, len(batched_sentences), sent_batch_size):
        chunk = batched_sentences[start : start + sent_batch_size]
        chunk_indices = indices[start : start + sent_batch_size]

        tok = tokzr(
            chunk,
            padding=True,
            truncation=True,
            max_length=truncation_max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        # with torch.no_grad():
        with torch.inference_mode():
            gen = model.generate(
                **tok,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        decoded = tokzr.batch_decode(gen, skip_special_tokens=True)

        for idx, t in zip(chunk_indices, decoded):
            t = (t or "").strip()
            if not t:
                continue
            outputs[idx] = (outputs[idx] + " " + t).strip() if outputs[idx] else t

    # fallback to original if empty
    for i in range(len(outputs)):
        if not outputs[i] and isinstance(texts[i], str):
            outputs[i] = texts[i]

    return outputs


def batch_translate_texts_patched_old(
    translator,
    texts,
    *,
    src_lang: str,
    tgt_lang: str,
    is_prompt=True,
    sent_batch_size=32,
    max_new_tokens=256,
    truncation_max_length=256,
):
    # split -> flatten
    batched_sentences, indices = [], []
    for i, text in enumerate(texts):
        for sent in split_sentences_bn(text, is_prompt=is_prompt):
            if sent.strip():
                batched_sentences.append(sent)
                indices.append(i)

    if not batched_sentences:
        return texts

    tokzr = translator.tokenizer
    model = translator.model
    device = model.device

    # ✅ NLLB requires these
    tokzr.src_lang = src_lang
    try:
        forced_bos = tokzr.lang_code_to_id[tgt_lang]
    except KeyError:
        raise KeyError(
            f"{tgt_lang} not in tokenizer.lang_code_to_id. "
            f"Double-check codes like eng_Latn, hau_Latn, ben_Beng."
        )

    decoded_all = []

    # ✅ actually use sent_batch_size
    for start in range(0, len(batched_sentences), sent_batch_size):
        chunk = batched_sentences[start : start + sent_batch_size]

        tok = tokzr(
            chunk,
            padding=True,
            truncation=True,
            max_length=truncation_max_length,  # input-side cap
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **tok,
                forced_bos_token_id=forced_bos,  # ✅ target language
                max_new_tokens=max_new_tokens,   # output-side cap
                num_beams=1,
                do_sample=False,
            )

        decoded_all.extend(tokzr.batch_decode(out, skip_special_tokens=True))

    # reconstruct per original text
    outputs = [""] * len(texts)
    for i, t in zip(indices, decoded_all):
        t = (t or "").strip()
        if t:
            outputs[i] = (outputs[i] + " " + t).strip() if outputs[i] else t

    # fallback to original if empty
    for i in range(len(outputs)):
        if not outputs[i] and isinstance(texts[i], str):
            outputs[i] = texts[i]

    return outputs

def _batch_translate_texts(
    translator,
    texts,
    *,
    is_prompt: bool,
    sent_batch_size: int,
    max_length: int,
    stats: Dict[str, int],
):
    """
    Sentence split -> flatten -> translate in sentence batches -> reconstruct per original text.
    """
    batched_sentences = []
    indices = []

    for idx, text in enumerate(texts):
        sents = _split_sentences(text)
        if len(sents) > 1:
            if is_prompt:
                stats["split_prompt_count"] += 1
            else:
                stats["split_response_count"] += 1

        for sent in sents:
            if sent.strip():
                batched_sentences.append(sent)
                indices.append(idx)

    if not batched_sentences:
        return texts

    translations = translator(
        batched_sentences,
        max_length=max_length,
        batch_size=sent_batch_size,
    )

    outputs = [""] * len(texts)
    for idx, trans in zip(indices, translations):
        t = trans.get("translation_text", "").strip()
        if not t:
            continue
        outputs[idx] = (outputs[idx] + " " + t).strip() if outputs[idx] else t

    # fallback: keep original if empty
    for i in range(len(outputs)):
        if not outputs[i] and isinstance(texts[i], str):
            outputs[i] = texts[i]

    return outputs


def translate(
    *,
    input_path: str,
    output_path: str,
    dataset: str = None,
    model_preset: str = "nllb200_600m",
    gpu_device: int = 0,
    obj_batch_size: int = 128,
    sent_batch_size: int = 32,
    max_new_tokens: int = 256,
    target_lang_key: Optional[str] = None,
    src_lang_override: Optional[str] = None,
    tgt_lang_override: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Translate fields `agg_prompt_bn` and `agg_response_bn` and inject:

      "translation": {
         "<lang_key>": {"prompt": "...", "response": "..."}
      }

    at the same level as agg_prompt_bn/agg_response_bn.

    Returns a dict of stats.
    """
    if model_preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model_preset={model_preset}. Available: {sorted(MODEL_PRESETS.keys())}"
        )

    preset = MODEL_PRESETS[model_preset]
    model_name = preset["model_name"]
    src_lang = src_lang_override or preset["src_lang"]
    tgt_lang = tgt_lang_override or preset["tgt_lang"]
    lang_key = target_lang_key or preset.get("lang_key", "sw")

    stats = {
        "model_preset": model_preset,
        "model_name": model_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "lang_key": lang_key,
        "gpu_device": gpu_device,
        "split_prompt_count": 0,
        "split_response_count": 0,
        "num_objects": 0,
    }

    translator = _build_translator(
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=gpu_device,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    stats["num_objects"] = len(data)

    num_batches = math.ceil(len(data) / obj_batch_size)

    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Translating ({src_lang}->{tgt_lang})", unit="batch")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")

        for b in iterator:
            batch_objs = data[b * obj_batch_size : (b + 1) * obj_batch_size]

            

            if dataset == "Aegis":
                prompts = [(obj.get("prompt") or "").strip() for obj in batch_objs]
                responses = [(obj.get("response") or "").strip() for obj in batch_objs]
            else:
                prompts = [(obj.get("agg_prompt_bn") or "").strip() for obj in batch_objs]
                responses = [(obj.get("agg_response_bn") or "").strip() for obj in batch_objs]

            prompts_tr = batch_translate_texts_patched(
                translator,
                prompts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
            )

            responses_tr = batch_translate_texts_patched(
                translator,
                responses,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                sent_batch_size=sent_batch_size,
                max_new_tokens=max_new_tokens,
                truncation_max_length=256,
                is_prompt=False,
            )

            for obj, p_t, r_t in zip(batch_objs, prompts_tr, responses_tr):
                if "translation" not in obj or not isinstance(obj["translation"], dict):
                    obj["translation"] = {}

                obj["translation"][lang_key] = {"prompt": p_t, "response": r_t}

            for i, obj in enumerate(batch_objs):
                json.dump(obj, f_out, ensure_ascii=False, indent=2)
                is_last = (b == num_batches - 1) and (i == len(batch_objs) - 1)
                f_out.write("\n" if is_last else ",\n")

            f_out.flush()

        f_out.write("]\n")

    stats["output_path"] = output_path
    return stats

