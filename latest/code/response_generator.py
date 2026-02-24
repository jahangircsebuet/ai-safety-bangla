import json
import argparse
import hashlib
import torch
import time
import transformers
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    # Prefer flash / mem-efficient kernels when available
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())


# -------------------------
# Response Generator (Gemma version)
# -------------------------
class ResponseGenerator:
    """
    This is the SAME code as your GemmaGeneratorFaster, only the class name is changed.

    Optimizations:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (avoid CPU/disk offload)
      - LEFT padding for decoder-only models
      - Render chat to strings then batch tokenize
      - Batch generate on GPU with autocast
      - batch_decode
      - returns generate() time per batch
      - prints hf_device_map (if present)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        compile_model: bool = True,
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        # pad_token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # sdpa
        )
        self.model.to(self.device)
        self.model.eval()

        # Optional compile (can speed up, sometimes no gain depending on env)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print("[WARN] torch.compile failed, continuing without compile:", repr(e))

        # Print hf_device_map if present (usually absent when not using accelerate)
        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time spent in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]
        print(f"device {next(self.model.parameters()).device}")

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + timing ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice + batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec



def generate_responses(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2-2b-it",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 32,
    max_input_tokens: int = 256,
    length_bucket: bool = True,
    save_every_batches: int = 20,
):
    import os
    import time
    import json
    from typing import Any, Dict, List, Optional, Tuple

    from tqdm import tqdm

    def _flush_json_atomic(path: str, obj: Any) -> None:
        """Atomic write to avoid corrupted output if interrupted."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
        },
    )

    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = ResponseGenerator(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        device="cuda:0",
        use_fast=True,
        trust_remote_code=False,
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing (reduce padding waste)
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved empty output.")
        return

    num_batches = (total + batch_size - 1) // batch_size
    processed = 0

    # Timing stats for model.generate()
    total_gen_time = 0.0
    gen_batches = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += gen_sec
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # Save every N batches (and last batch)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)

            elapsed = time.time() - t_start
            avg_gen = total_gen_time / max(1, gen_batches)
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    elapsed = time.time() - t_start
    avg_gen = total_gen_time / max(1, gen_batches)
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")


# -------------------------
# Gemma Generator (Optimized: sdpa, no offload, batch decode, timing)
# -------------------------
class GemmaGeneratorFaster:
    """
    Optimizations:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (avoid CPU/disk offload)
      - LEFT padding for decoder-only models
      - Render chat to strings then batch tokenize
      - Batch generate on GPU with autocast
      - batch_decode
      - returns generate() time per batch
      - prints hf_device_map (if present)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        compile_model: bool = True,
    ):
        
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        # pad_token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # sdpa
        )
        self.model.to(self.device)
        self.model.eval()
        # Optional compile (can speed up, sometimes no gain depending on env)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print("[WARN] torch.compile failed, continuing without compile:", repr(e))

        # Print hf_device_map if present (usually absent when not using accelerate)
        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time spent in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]
        print(f"device {next(self.model.parameters()).device}")

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + timing ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice + batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec


# -------------------------
# Main callable function (Optimized like your Llama version)
# -------------------------
def generate_gemma_responses(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2-2b-it",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,          # << per your speed settings
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 32,              # << match your new default
    max_input_tokens: int = 256,       # << per your speed settings
    length_bucket: bool = True,
    save_every_batches: int = 20,      # << save every N batches
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
        },
    )

    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = GemmaGeneratorFaster(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        # attn_implementation="flash_attention_2",
        device="cuda:0",
        use_fast=True,
        trust_remote_code=False,
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing (reduce padding waste)
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved empty output.")
        return

    num_batches = (total + batch_size - 1) // batch_size
    processed = 0

    # Timing stats for model.generate()
    total_gen_time = 0.0
    gen_batches = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += gen_sec
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # Save every N batches (and last batch)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)

            elapsed = time.time() - t_start
            avg_gen = total_gen_time / max(1, gen_batches)
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    t_end = time.time()
    elapsed = t_end - t_start
    avg_gen = total_gen_time / max(1, gen_batches)
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")


# -------------------------
# Gemma Generator (batched + faster)
# -------------------------
class GemmaGeneratorFasterV1:
    """
    Works with Gemma instruction-tuned models (suffix: -it), e.g.:
      - "google/gemma-2-2b-it"
      - "google/gemma-2-9b-it"
      - "google/gemma-7b-it"
      - "google/gemma-3-1b-it" (if available in your environment)

    Optimizations:
      - Render chat template to strings then batch tokenize
      - Batch generate on GPU
      - Autocast BF16/FP16
      - TF32 enabled (Ampere+)
      - Optional length bucketing (outside)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        # permission issue
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Gemma-it supports chat templates in recent transformers
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]
        print(f"device {next(self.model.parameters()).device}")

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)
        
        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return texts


# -------------------------
# Main callable function
# -------------------------
def generate_gemma_responses_v1(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2-2b-it",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )
    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = GemmaGeneratorFaster(
        model_name,
        dtype=dtype,
        device_map="auto",
        use_fast=True,
        trust_remote_code=False,
    )

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100
        print(f"[Progress] {processed}/{total} ({percent:.2f}%)")
        t_end = time.time()
        elapsed_sec = t_end - t_start
        print(f"Total time taken so far: {elapsed_sec:.2f} seconds")

    final_out = [x for x in out if x is not None]
    write_json(output_path, final_out)
    print(f"Saved: {output_path}")

    # --- end timing (right after saving JSON) ---
    t_end = time.time()
    elapsed_sec = t_end - t_start
    print(f"Total time (load -> save): {elapsed_sec:.2f} seconds")


# -------------------------
# Mistral/Mixtral Generator (Optimized)
# -------------------------
class MistralMixtralGeneratorFaster:
    """
    Optimizations:
      - Batch chat-template rendering (tokenize=False) then batch tokenize
      - Batch generate on GPU
      - Autocast (BF16/FP16)
      - TF32 enabled (Ampere+)
      - Optional FlashAttention2 via attn_implementation
      - Length bucketing (optional, done outside)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        self.model_name = str(model_name)

        # Optional: small speed boost on Ampere+
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        
        # put gemma config - check if it is faster 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )

        # mistral earlier config 
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=use_fast)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     device_map=device_map,
        #     dtype=dtype,   # dtype (not torch_dtype)
        # )
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        self.autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    def _render_chat(self, prompt: str, system: str) -> str:
        """
        Render a single chat prompt string using the model's chat template.
        Rendering text first lets us batch-tokenize efficiently.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Important: tokenize=False returns a string
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,  # truncate input to avoid huge CPU/GPU work
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        gen_texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return gen_texts


# -------------------------
# Main callable function (Optimized)
# -------------------------
def generate_mistral_mixtral_faster(
    input_path: str,
    output_path: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )

    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    # ✅ Use your optimized class
    gen = MistralMixtralGeneratorFaster(
        model_name,
        dtype=dtype,
        device_map="auto",
        use_fast=True,
        trust_remote_code=False,
    )

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100
        print(f"[Progress] {processed}/{total} ({percent:.2f}%)")
        t_end = time.time()
        elapsed_sec = t_end - t_start
        print(f"Total time taken so far: {elapsed_sec:.2f} seconds")

    final_out = [x for x in out if x is not None]
    write_json(output_path, final_out)
    print(f"Saved: {output_path}")

    # --- end timing (right after saving JSON) ---
    t_end = time.time()
    elapsed_sec = t_end - t_start
    print(f"Total time (load -> save): {elapsed_sec:.2f} seconds")



import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Utils
# -------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flush_json_atomic(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -------------------------
# Qwen Generator (Optimized: sdpa, no offload, batch decode, timing)
# -------------------------
class QwenGeneratorFaster:
    """
    Optimizations:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (avoid CPU/disk offload)
      - LEFT padding for decoder-only models
      - Render chat to strings then batch tokenize
      - Batch generate on GPU with autocast
      - batch_decode
      - returns generate() time per batch
      - prints hf_device_map (if present)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        compile_model: bool = True,
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs -> left padding for correct batched generation
        self.tokenizer.padding_side = "left"

        # pad_token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # "sdpa"
        )
        self.model.to(self.device)
        self.model.eval()

        # Optional compile (can speed up, sometimes no gain depending on env)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print("[WARN] torch.compile failed, continuing without compile:", repr(e))

        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
        pad_to_multiple_of: int = 8,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + time ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice + batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec


# -------------------------
# Main callable function (Optimized like Gemma)
# -------------------------
def generate_qwen_responses(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 16,
    max_input_tokens: int = 256,
    length_bucket: bool = True,
    save_every_batches: int = 20,
    attn_implementation: str = "sdpa",
    device: str = "cuda:0",
    compile_model: bool = True,
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
            "attn_implementation": attn_implementation,
            "device": device,
            "compile_model": compile_model,
        },
    )

    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = QwenGeneratorFaster(
        model_name,
        dtype=dtype,
        use_fast=True,
        trust_remote_code=False,
        attn_implementation=attn_implementation,
        device=device,
        compile_model=compile_model,
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out[idx] = {id_key: obj_id, prompt_key: prompt, "response": "", "error": "missing_prompt"}
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out[idx] = {id_key: obj_id, prompt_key: prompt, "response": "", "deduped": True, "dedup_key": h}
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved.")
        return

    num_batches = (total + batch_size - 1) // batch_size

    # timing stats for model.generate()
    total_gen_time = 0.0
    gen_batches = 0
    processed = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += float(gen_sec)
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # checkpoint save every N batches (and last)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)
            elapsed = time.time() - t_start
            avg_gen = total_gen_time / max(1, gen_batches)
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    elapsed = time.time() - t_start
    avg_gen = total_gen_time / max(1, gen_batches)
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")



# -------------------------
# Qwen Generator
# -------------------------
class QwenGeneratorV1:
    """
    Qwen Instruct generator using chat template.
    Works with Qwen2 / Qwen2.5 Instruct models.

    Example model names:
      - "Qwen/Qwen2.5-7B-Instruct"
      - "Qwen/Qwen2.5-14B-Instruct"
      - "Qwen/Qwen2-7B-Instruct"
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        self.model_name = str(model_name)
        # Optional: small speed boost on Ampere+
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,     # accelerate placement
            dtype=dtype,               # NOTE: use dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Ensure pad_token_id exists for generation
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )


    def _render_chat(self, prompt: str, system: str) -> str:
        """
        Render a single chat prompt string using the model's chat template.
        Rendering text first lets us batch-tokenize efficiently.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Important: tokenize=False returns a string
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,  # truncate input to avoid huge CPU/GPU work
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        gen_texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return gen_texts
    
    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# -------------------------
# Main callable function
# -------------------------
def generate_qwen(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    model_name = str(model_name)
    prompt_key = str(prompt_key)
    id_key = str(id_key)

    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )

    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    # ✅ Use your optimized class
    gen = QwenGenerator(model_name, dtype=dtype, device_map="auto", use_fast=True, trust_remote_code=False)

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100
        print(f"[Progress] {processed}/{total} ({percent:.2f}%)")
        t_end = time.time()
        elapsed_sec = t_end - t_start
        print(f"Total time taken so far: {elapsed_sec:.2f} seconds")

    final_out = [x for x in out if x is not None]
    write_json(output_path, final_out)
    print(f"Saved: {output_path}")

    # --- end timing (right after saving JSON) ---
    t_end = time.time()
    elapsed_sec = t_end - t_start
    print(f"Total time (load -> save): {elapsed_sec:.2f} seconds")

# -------------------------
# Utils
# -------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Llama Generator
# -------------------------
class LlamaGenerator_slow:
    """
    Minimal chat-style generator for instruct models (LLaMA-3, Qwen, etc.)
    using tokenizer.apply_chat_template().
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Some tokenizers don't have pad_token set; for generation it's often safe to reuse eos.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = self.model.device
        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
    def _render_chat(self, prompt: str, system: str) -> str:
        """
        Render a single chat prompt string using the model's chat template.
        Rendering text first lets us batch-tokenize efficiently.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Important: tokenize=False returns a string
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,  # truncate input to avoid huge CPU/GPU work
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        gen_texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return gen_texts
    
    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return text


# -------------------------
# Main
# -------------------------

def generate_llama_slow(
    input_path: str,
    output_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    import os
    import time
    import json
    from typing import Any, Dict, List, Optional, Tuple

    def _flush_json_atomic(path: str, obj: Any) -> None:
        """Atomic write to avoid corrupted output if interrupted."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )

    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root to be a list, got {type(data).__name__}")

    # ✅ Use your optimized class
    gen = LlamaGenerator(model_name)

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    # ✅ If output exists, we can resume (optional but useful)
    if os.path.exists(output_path):
        try:
            existing = load_json(output_path)
            if isinstance(existing, list) and len(existing) > 0:
                # map existing entries back into `out` by id if possible
                existing_by_id = {str(e.get(id_key)): e for e in existing if isinstance(e, dict) and e.get(id_key) is not None}
                for i, obj in enumerate(data):
                    oid = obj.get(id_key)
                    if oid is None:
                        continue
                    key = str(oid)
                    if key in existing_by_id:
                        out[i] = existing_by_id[key]
                print(f"[Resume] Loaded existing output: {output_path}")
        except Exception:
            pass

    # --- Batched generation loop ---
    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        # Skip batch if already filled (supports resume / partial runs)
        if all(out[i] is not None for i in idxs):
            processed += len(batch_items)
            percent = (processed / max(1, total)) * 100
            print(f"[Skip] batch already done. {processed}/{total} ({percent:.2f}%)")
            continue

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100

        # ✅ SAVE AFTER EACH BATCH
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)

        t_now = time.time()
        elapsed_sec = t_now - t_start
        print(f"[Progress] {processed}/{total} ({percent:.2f}%) | elapsed={elapsed_sec:.2f}s | saved={output_path}")

    # Final save (already saved per-batch, but keep for safety)
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)
    print(f"Saved (final): {output_path}")

    t_end = time.time()
    print(f"Total time (load -> final save): {t_end - t_start:.2f} seconds")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Utils
# -------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flush_json_atomic(path: str, obj: Any) -> None:
    """Atomic write to avoid corrupted output if interrupted."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -------------------------
# Llama Generator (Optimized)
# -------------------------
class LlamaGenerator:
    """
    Optimizations included:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (no offload)
      - left padding
      - batch rendering + batch tokenize + batch generate
      - batch_decode
      - prints hf_device_map (if present)
      - returns generate() time for per-batch logging
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        # Some tokenizers don't have pad_token set; for generation it's often safe to reuse eos.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # sdpa
        )
        self.model.to(self.device)
        self.model.eval()

        # Print hf_device_map if present (usually absent when not using accelerate)
        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time spent in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + timing ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice generated continuations then batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec


# -------------------------
# Main (save every N batches, tqdm, batch decode, no gen_params in output)
# -------------------------
def generate_llama(
    input_path: str,
    output_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_key: str = "agg_prompt_bn",   # matches your input structure
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,           # per requirement
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 16,                # per requirement
    max_input_tokens: int = 256,         # per requirement
    length_bucket: bool = True,
    save_every_batches: int = 10,        # per requirement
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
        },
    )

    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root to be a list, got {type(data).__name__}")

    # ✅ No offload: device_map=None + model.to("cuda:0") inside generator
    gen = LlamaGenerator(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        device="cuda:0",
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved empty output.")
        return

    num_batches = (total + batch_size - 1) // batch_size
    processed = 0

    # Stats for logging
    total_gen_time = 0.0
    gen_batches = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += gen_sec
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # ✅ Save every N batches (and at the end)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)

            elapsed = time.time() - t_start
            avg_gen = (total_gen_time / max(1, gen_batches))
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    t_end = time.time()
    elapsed = t_end - t_start
    avg_gen = (total_gen_time / max(1, gen_batches))
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="Input JSON file (list of objects).")
#     parser.add_argument("--output", required=True, help="Output JSON file.")
#     parser.add_argument(
#         "--model-name",
#         default="meta-llama/Meta-Llama-3-8B-Instruct",
#         help="HF model name/path for response generation.",
#     )

#     # which field to read as the prompt
#     parser.add_argument("--prompt-key", default="prompt_bn", help="Field name holding the prompt text.")
#     parser.add_argument("--id-key", default="id", help="Field name holding the id.")

#     # generation params
#     parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt.")
#     parser.add_argument("--max-new-tokens", type=int, default=512)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--top-p", type=float, default=0.9)
#     parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (no sampling).")

#     # behavior
#     parser.add_argument("--dedup", action="store_true", help="Deduplicate identical prompt texts.")
#     parser.add_argument("--keep-input-fields", action="store_true",
#                         help="If set, copy full original obj into output entry as `input_obj`.")

#     args = parser.parse_args()
#     print("args:", args)

#     data = load_json(args.input)
#     if not isinstance(data, list):
#         raise ValueError(f"Expected JSON root to be a list, got {type(data).__name__}")

#     gen = LLMGenerator(args.model_name)

#     seen = set()
#     out: List[Dict[str, Any]] = []

#     total = len(data)
#     processed = 0

#     for obj in data:
#         if not isinstance(obj, dict):
#             continue

#         obj_id = obj.get(args.id_key)
#         prompt = str(obj.get(args.prompt_key, "")).strip()

#         # If prompt missing, still write stub entry
#         if not prompt:
#             out_entry: Dict[str, Any] = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": "",
#                 "error": "missing_prompt",
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)
#             continue

#         h = sha1(prompt)
#         if args.dedup and h in seen:
#             # Keep an entry noting it was deduped (so ids are preserved)
#             out_entry = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": "",
#                 "deduped": True,
#                 "dedup_key": h,
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)
#             continue
#         seen.add(h)

#         try:
#             response = gen.generate_one(
#                 prompt,
#                 system=args.system,
#                 max_new_tokens=args.max_new_tokens,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 do_sample=(not args.greedy),
#             )
#             out_entry = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": response,
#                 "model": args.model_name,
#                 "gen_params": {
#                     "max_new_tokens": args.max_new_tokens,
#                     "temperature": args.temperature,
#                     "top_p": args.top_p,
#                     "greedy": bool(args.greedy),
#                 },
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)

#         except Exception as e:
#             out_entry = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": "",
#                 "error": repr(e),
#                 "model": args.model_name,
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)

#         processed += 1
#         percent = (processed / total) * 100
#         print(f"[Progress] {processed}/{total} ({percent:.2f}%)")

#     write_json(args.output, out)
#     print(f"Saved: {args.output}")import json
import argparse
import hashlib
import torch
import time
import transformers
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    # Prefer flash / mem-efficient kernels when available
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())


# -------------------------
# Response Generator (Gemma version)
# -------------------------
class ResponseGenerator:
    """
    This is the SAME code as your GemmaGeneratorFaster, only the class name is changed.

    Optimizations:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (avoid CPU/disk offload)
      - LEFT padding for decoder-only models
      - Render chat to strings then batch tokenize
      - Batch generate on GPU with autocast
      - batch_decode
      - returns generate() time per batch
      - prints hf_device_map (if present)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        compile_model: bool = True,
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        # pad_token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # sdpa
        )
        self.model.to(self.device)
        self.model.eval()

        # Optional compile (can speed up, sometimes no gain depending on env)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print("[WARN] torch.compile failed, continuing without compile:", repr(e))

        # Print hf_device_map if present (usually absent when not using accelerate)
        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time spent in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]
        print(f"device {next(self.model.parameters()).device}")

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + timing ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice + batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec



def generate_responses(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2-2b-it",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 32,
    max_input_tokens: int = 256,
    length_bucket: bool = True,
    save_every_batches: int = 20,
):
    import os
    import time
    import json
    from typing import Any, Dict, List, Optional, Tuple

    from tqdm import tqdm

    def _flush_json_atomic(path: str, obj: Any) -> None:
        """Atomic write to avoid corrupted output if interrupted."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
        },
    )

    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = ResponseGenerator(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        device="cuda:0",
        use_fast=True,
        trust_remote_code=False,
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing (reduce padding waste)
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved empty output.")
        return

    num_batches = (total + batch_size - 1) // batch_size
    processed = 0

    # Timing stats for model.generate()
    total_gen_time = 0.0
    gen_batches = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += gen_sec
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # Save every N batches (and last batch)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)

            elapsed = time.time() - t_start
            avg_gen = total_gen_time / max(1, gen_batches)
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    elapsed = time.time() - t_start
    avg_gen = total_gen_time / max(1, gen_batches)
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")


# -------------------------
# Gemma Generator (Optimized: sdpa, no offload, batch decode, timing)
# -------------------------
class GemmaGeneratorFaster:
    """
    Optimizations:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (avoid CPU/disk offload)
      - LEFT padding for decoder-only models
      - Render chat to strings then batch tokenize
      - Batch generate on GPU with autocast
      - batch_decode
      - returns generate() time per batch
      - prints hf_device_map (if present)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        compile_model: bool = True,
    ):
        
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        # pad_token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # sdpa
        )
        self.model.to(self.device)
        self.model.eval()
        # Optional compile (can speed up, sometimes no gain depending on env)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print("[WARN] torch.compile failed, continuing without compile:", repr(e))

        # Print hf_device_map if present (usually absent when not using accelerate)
        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time spent in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]
        print(f"device {next(self.model.parameters()).device}")

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + timing ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice + batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec


# -------------------------
# Main callable function (Optimized like your Llama version)
# -------------------------
def generate_gemma_responses(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2-2b-it",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,          # << per your speed settings
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 32,              # << match your new default
    max_input_tokens: int = 256,       # << per your speed settings
    length_bucket: bool = True,
    save_every_batches: int = 20,      # << save every N batches
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
        },
    )

    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = GemmaGeneratorFaster(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        # attn_implementation="flash_attention_2",
        device="cuda:0",
        use_fast=True,
        trust_remote_code=False,
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing (reduce padding waste)
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved empty output.")
        return

    num_batches = (total + batch_size - 1) // batch_size
    processed = 0

    # Timing stats for model.generate()
    total_gen_time = 0.0
    gen_batches = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += gen_sec
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # Save every N batches (and last batch)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)

            elapsed = time.time() - t_start
            avg_gen = total_gen_time / max(1, gen_batches)
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    t_end = time.time()
    elapsed = t_end - t_start
    avg_gen = total_gen_time / max(1, gen_batches)
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")


# -------------------------
# Gemma Generator (batched + faster)
# -------------------------
class GemmaGeneratorFasterV1:
    """
    Works with Gemma instruction-tuned models (suffix: -it), e.g.:
      - "google/gemma-2-2b-it"
      - "google/gemma-2-9b-it"
      - "google/gemma-7b-it"
      - "google/gemma-3-1b-it" (if available in your environment)

    Optimizations:
      - Render chat template to strings then batch tokenize
      - Batch generate on GPU
      - Autocast BF16/FP16
      - TF32 enabled (Ampere+)
      - Optional length bucketing (outside)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        # permission issue
        # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Gemma-it supports chat templates in recent transformers
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]
        print(f"device {next(self.model.parameters()).device}")

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)
        
        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return texts


# -------------------------
# Main callable function
# -------------------------
def generate_gemma_responses_v1(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2-2b-it",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )
    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = GemmaGeneratorFaster(
        model_name,
        dtype=dtype,
        device_map="auto",
        use_fast=True,
        trust_remote_code=False,
    )

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100
        print(f"[Progress] {processed}/{total} ({percent:.2f}%)")
        t_end = time.time()
        elapsed_sec = t_end - t_start
        print(f"Total time taken so far: {elapsed_sec:.2f} seconds")

    final_out = [x for x in out if x is not None]
    write_json(output_path, final_out)
    print(f"Saved: {output_path}")

    # --- end timing (right after saving JSON) ---
    t_end = time.time()
    elapsed_sec = t_end - t_start
    print(f"Total time (load -> save): {elapsed_sec:.2f} seconds")


# -------------------------
# Mistral/Mixtral Generator (Optimized)
# -------------------------
class MistralMixtralGeneratorFaster:
    """
    Optimizations:
      - Batch chat-template rendering (tokenize=False) then batch tokenize
      - Batch generate on GPU
      - Autocast (BF16/FP16)
      - TF32 enabled (Ampere+)
      - Optional FlashAttention2 via attn_implementation
      - Length bucketing (optional, done outside)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        self.model_name = str(model_name)

        # Optional: small speed boost on Ampere+
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        
        # put gemma config - check if it is faster 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )

        # mistral earlier config 
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=use_fast)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     device_map=device_map,
        #     dtype=dtype,   # dtype (not torch_dtype)
        # )
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        self.autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    def _render_chat(self, prompt: str, system: str) -> str:
        """
        Render a single chat prompt string using the model's chat template.
        Rendering text first lets us batch-tokenize efficiently.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Important: tokenize=False returns a string
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,  # truncate input to avoid huge CPU/GPU work
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        gen_texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return gen_texts


# -------------------------
# Main callable function (Optimized)
# -------------------------
def generate_mistral_mixtral_faster(
    input_path: str,
    output_path: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )

    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    # ✅ Use your optimized class
    gen = MistralMixtralGeneratorFaster(
        model_name,
        dtype=dtype,
        device_map="auto",
        use_fast=True,
        trust_remote_code=False,
    )

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100
        print(f"[Progress] {processed}/{total} ({percent:.2f}%)")
        t_end = time.time()
        elapsed_sec = t_end - t_start
        print(f"Total time taken so far: {elapsed_sec:.2f} seconds")

    final_out = [x for x in out if x is not None]
    write_json(output_path, final_out)
    print(f"Saved: {output_path}")

    # --- end timing (right after saving JSON) ---
    t_end = time.time()
    elapsed_sec = t_end - t_start
    print(f"Total time (load -> save): {elapsed_sec:.2f} seconds")



import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Utils
# -------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flush_json_atomic(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -------------------------
# Qwen Generator (Optimized: sdpa, no offload, batch decode, timing)
# -------------------------
class QwenGeneratorFaster:
    """
    Optimizations:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (avoid CPU/disk offload)
      - LEFT padding for decoder-only models
      - Render chat to strings then batch tokenize
      - Batch generate on GPU with autocast
      - batch_decode
      - returns generate() time per batch
      - prints hf_device_map (if present)
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
        compile_model: bool = True,
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs -> left padding for correct batched generation
        self.tokenizer.padding_side = "left"

        # pad_token safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # "sdpa"
        )
        self.model.to(self.device)
        self.model.eval()

        # Optional compile (can speed up, sometimes no gain depending on env)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print("[WARN] torch.compile failed, continuing without compile:", repr(e))

        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
        pad_to_multiple_of: int = 8,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + time ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice + batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec


# -------------------------
# Main callable function (Optimized like Gemma)
# -------------------------
def generate_qwen_responses(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 16,
    max_input_tokens: int = 256,
    length_bucket: bool = True,
    save_every_batches: int = 20,
    attn_implementation: str = "sdpa",
    device: str = "cuda:0",
    compile_model: bool = True,
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
            "attn_implementation": attn_implementation,
            "device": device,
            "compile_model": compile_model,
        },
    )

    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    gen = QwenGeneratorFaster(
        model_name,
        dtype=dtype,
        use_fast=True,
        trust_remote_code=False,
        attn_implementation=attn_implementation,
        device=device,
        compile_model=compile_model,
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out[idx] = {id_key: obj_id, prompt_key: prompt, "response": "", "error": "missing_prompt"}
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out[idx] = {id_key: obj_id, prompt_key: prompt, "response": "", "deduped": True, "dedup_key": h}
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved.")
        return

    num_batches = (total + batch_size - 1) // batch_size

    # timing stats for model.generate()
    total_gen_time = 0.0
    gen_batches = 0
    processed = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += float(gen_sec)
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # checkpoint save every N batches (and last)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)
            elapsed = time.time() - t_start
            avg_gen = total_gen_time / max(1, gen_batches)
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    elapsed = time.time() - t_start
    avg_gen = total_gen_time / max(1, gen_batches)
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")



# -------------------------
# Qwen Generator
# -------------------------
class QwenGeneratorV1:
    """
    Qwen Instruct generator using chat template.
    Works with Qwen2 / Qwen2.5 Instruct models.

    Example model names:
      - "Qwen/Qwen2.5-7B-Instruct"
      - "Qwen/Qwen2.5-14B-Instruct"
      - "Qwen/Qwen2-7B-Instruct"
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        self.model_name = str(model_name)
        # Optional: small speed boost on Ampere+
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,     # accelerate placement
            dtype=dtype,               # NOTE: use dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Ensure pad_token_id exists for generation
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )


    def _render_chat(self, prompt: str, system: str) -> str:
        """
        Render a single chat prompt string using the model's chat template.
        Rendering text first lets us batch-tokenize efficiently.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Important: tokenize=False returns a string
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,  # truncate input to avoid huge CPU/GPU work
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        gen_texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return gen_texts
    
    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# -------------------------
# Main callable function
# -------------------------
def generate_qwen(
    input_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    model_name = str(model_name)
    prompt_key = str(prompt_key)
    id_key = str(id_key)

    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )

    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list in {input_path}, got {type(data).__name__}")

    # ✅ Use your optimized class
    gen = QwenGenerator(model_name, dtype=dtype, device_map="auto", use_fast=True, trust_remote_code=False)

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100
        print(f"[Progress] {processed}/{total} ({percent:.2f}%)")
        t_end = time.time()
        elapsed_sec = t_end - t_start
        print(f"Total time taken so far: {elapsed_sec:.2f} seconds")

    final_out = [x for x in out if x is not None]
    write_json(output_path, final_out)
    print(f"Saved: {output_path}")

    # --- end timing (right after saving JSON) ---
    t_end = time.time()
    elapsed_sec = t_end - t_start
    print(f"Total time (load -> save): {elapsed_sec:.2f} seconds")

# -------------------------
# Utils
# -------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Llama Generator
# -------------------------
class LlamaGenerator_slow:
    """
    Minimal chat-style generator for instruct models (LLaMA-3, Qwen, etc.)
    using tokenizer.apply_chat_template().
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Some tokenizers don't have pad_token set; for generation it's often safe to reuse eos.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = self.model.device
        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
    def _render_chat(self, prompt: str, system: str) -> str:
        """
        Render a single chat prompt string using the model's chat template.
        Rendering text first lets us batch-tokenize efficiently.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Important: tokenize=False returns a string
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 1024,  # truncate input to avoid huge CPU/GPU work
    ) -> List[str]:
        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1)

        # 3) Batch generate (GPU)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # 4) Slice generated continuations and decode in batch
        # outputs shape: [B, seq_len_total]
        gen_texts: List[str] = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i].item())
            gen_ids = outputs[i, start:]
            gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        return gen_texts
    
    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return text


# -------------------------
# Main
# -------------------------

def generate_llama_slow(
    input_path: str,
    output_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_key: str = "prompt_bn",
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 8,
    max_input_tokens: int = 1024,
    length_bucket: bool = True,
):
    import os
    import time
    import json
    from typing import Any, Dict, List, Optional, Tuple

    def _flush_json_atomic(path: str, obj: Any) -> None:
        """Atomic write to avoid corrupted output if interrupted."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
        },
    )

    # --- start timing (right before loading JSON) ---
    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root to be a list, got {type(data).__name__}")

    # ✅ Use your optimized class
    gen = LlamaGenerator(model_name)

    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    processed = 0

    # ✅ If output exists, we can resume (optional but useful)
    if os.path.exists(output_path):
        try:
            existing = load_json(output_path)
            if isinstance(existing, list) and len(existing) > 0:
                # map existing entries back into `out` by id if possible
                existing_by_id = {str(e.get(id_key)): e for e in existing if isinstance(e, dict) and e.get(id_key) is not None}
                for i, obj in enumerate(data):
                    oid = obj.get(id_key)
                    if oid is None:
                        continue
                    key = str(oid)
                    if key in existing_by_id:
                        out[i] = existing_by_id[key]
                print(f"[Resume] Loaded existing output: {output_path}")
        except Exception:
            pass

    # --- Batched generation loop ---
    for start in range(0, total, batch_size):
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        # Skip batch if already filled (supports resume / partial runs)
        if all(out[i] is not None for i in idxs):
            processed += len(batch_items)
            percent = (processed / max(1, total)) * 100
            print(f"[Skip] batch already done. {processed}/{total} ({percent:.2f}%)")
            continue

        try:
            responses = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "gen_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "greedy": bool(greedy),
                        "batch_size": batch_size,
                        "max_input_tokens": max_input_tokens,
                    },
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)
        percent = (processed / max(1, total)) * 100

        # ✅ SAVE AFTER EACH BATCH
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)

        t_now = time.time()
        elapsed_sec = t_now - t_start
        print(f"[Progress] {processed}/{total} ({percent:.2f}%) | elapsed={elapsed_sec:.2f}s | saved={output_path}")

    # Final save (already saved per-batch, but keep for safety)
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)
    print(f"Saved (final): {output_path}")

    t_end = time.time()
    print(f"Total time (load -> final save): {t_end - t_start:.2f} seconds")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Utils
# -------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flush_json_atomic(path: str, obj: Any) -> None:
    """Atomic write to avoid corrupted output if interrupted."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -------------------------
# Llama Generator (Optimized)
# -------------------------
class LlamaGenerator:
    """
    Optimizations included:
      - attn_implementation="sdpa"
      - device_map=None, model.to("cuda:0") (no offload)
      - left padding
      - batch rendering + batch tokenize + batch generate
      - batch_decode
      - prints hf_device_map (if present)
      - returns generate() time for per-batch logging
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str = "sdpa",
        device: str = "cuda:0",
    ):
        self.model_name = str(model_name)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )

        # decoder-only LMs should use LEFT padding for batched generation
        self.tokenizer.padding_side = "left"

        # Some tokenizers don't have pad_token set; for generation it's often safe to reuse eos.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Force single GPU placement (no accelerate sharding/offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,  # dtype (not torch_dtype)
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,  # sdpa
        )
        self.model.to(self.device)
        self.model.eval()

        # Print hf_device_map if present (usually absent when not using accelerate)
        if hasattr(self.model, "hf_device_map"):
            print("hf_device_map:", self.model.hf_device_map)
        else:
            print("hf_device_map: (not available; model loaded without accelerate)")

        self.autocast_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def _render_chat(self, prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        *,
        system: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_input_tokens: int = 256,
    ) -> Tuple[List[str], float]:
        """
        Returns (decoded_texts, generate_seconds) where generate_seconds is ONLY time spent in model.generate().
        """

        # 1) Render chat strings (CPU)
        chat_texts = [self._render_chat(p, system) for p in prompts]

        # 2) Batch tokenize (CPU -> single transfer to GPU)
        enc = self.tokenizer(
            chat_texts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

        # Track per-sample prompt lengths to slice generated part efficiently
        input_lens = attention_mask.sum(dim=1).tolist()

        # 3) Batch generate (GPU) + timing ONLY generate()
        t0 = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_sec = time.time() - t0

        # 4) Slice generated continuations then batch decode
        gen_ids_list = []
        for i in range(outputs.size(0)):
            start = int(input_lens[i])
            gen_ids_list.append(outputs[i, start:])

        texts = self.tokenizer.batch_decode(gen_ids_list, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        return texts, gen_sec


# -------------------------
# Main (save every N batches, tqdm, batch decode, no gen_params in output)
# -------------------------
def generate_llama(
    input_path: str,
    output_path: str,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_key: str = "agg_prompt_bn",   # matches your input structure
    id_key: str = "id",
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,           # per requirement
    temperature: float = 0.7,
    top_p: float = 0.9,
    greedy: bool = False,
    dedup: bool = False,
    keep_input_fields: bool = False,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 16,                # per requirement
    max_input_tokens: int = 256,         # per requirement
    length_bucket: bool = True,
    save_every_batches: int = 10,        # per requirement
):
    print(
        "config:",
        {
            "input_path": input_path,
            "output_path": output_path,
            "model_name": model_name,
            "prompt_key": prompt_key,
            "id_key": id_key,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": greedy,
            "dedup": dedup,
            "keep_input_fields": keep_input_fields,
            "dtype": str(dtype),
            "batch_size": batch_size,
            "max_input_tokens": max_input_tokens,
            "length_bucket": length_bucket,
            "save_every_batches": save_every_batches,
        },
    )

    t_start = time.time()

    print("Loading data....")
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root to be a list, got {type(data).__name__}")

    # ✅ No offload: device_map=None + model.to("cuda:0") inside generator
    gen = LlamaGenerator(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        device="cuda:0",
    )

    # Build work list
    seen = set()
    work: List[Tuple[int, Dict[str, Any], str]] = []  # (index, obj, prompt)
    out: List[Optional[Dict[str, Any]]] = [None] * len(data)

    for idx, obj in enumerate(data):
        if not isinstance(obj, dict):
            out[idx] = {"error": "non_dict_item", "index": idx}
            continue

        obj_id = obj.get(id_key)
        prompt = str(obj.get(prompt_key, "")).strip()

        if not prompt:
            out_entry: Dict[str, Any] = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "error": "missing_prompt",
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue

        h = sha1(prompt)
        if dedup and h in seen:
            out_entry = {
                id_key: obj_id,
                prompt_key: prompt,
                "response": "",
                "deduped": True,
                "dedup_key": h,
            }
            if keep_input_fields:
                out_entry["input_obj"] = obj
            out[idx] = out_entry
            continue
        seen.add(h)

        work.append((idx, obj, prompt))

    # Optional length bucketing to reduce padding
    if length_bucket and len(work) > 1:
        def _len_est(p: str) -> int:
            return len(gen.tokenizer(p, add_special_tokens=False).input_ids)
        work.sort(key=lambda x: _len_est(x[2]))

    total = len(work)
    if total == 0:
        final_out = [x for x in out if x is not None]
        _flush_json_atomic(output_path, final_out)
        print("No prompts to process. Saved empty output.")
        return

    num_batches = (total + batch_size - 1) // batch_size
    processed = 0

    # Stats for logging
    total_gen_time = 0.0
    gen_batches = 0

    for b in tqdm(range(num_batches), desc="Generating", unit="batch"):
        start = b * batch_size
        batch_items = work[start : start + batch_size]
        idxs = [it[0] for it in batch_items]
        objs = [it[1] for it in batch_items]
        prompts = [it[2] for it in batch_items]

        try:
            responses, gen_sec = gen.generate_batch(
                prompts,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(not greedy),
                max_input_tokens=max_input_tokens,
            )
            total_gen_time += gen_sec
            gen_batches += 1

            for orig_idx, obj, prompt, resp in zip(idxs, objs, prompts, responses):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": resp,
                    "model": model_name,
                    "batch_generate_seconds": round(float(gen_sec), 4),
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        except Exception as e:
            for orig_idx, obj, prompt in zip(idxs, objs, prompts):
                out_entry = {
                    id_key: obj.get(id_key),
                    prompt_key: prompt,
                    "response": "",
                    "error": repr(e),
                    "model": model_name,
                }
                if keep_input_fields:
                    out_entry["input_obj"] = obj
                out[orig_idx] = out_entry

        processed += len(batch_items)

        # ✅ Save every N batches (and at the end)
        if (b + 1) % save_every_batches == 0 or (b + 1) == num_batches:
            final_out = [x for x in out if x is not None]
            _flush_json_atomic(output_path, final_out)

            elapsed = time.time() - t_start
            avg_gen = (total_gen_time / max(1, gen_batches))
            tqdm.write(
                f"[Checkpoint] batches={b+1}/{num_batches} | items={processed}/{total} "
                f"| elapsed={elapsed:.1f}s | avg_generate_per_batch={avg_gen:.3f}s | saved={output_path}"
            )

    # Final save
    final_out = [x for x in out if x is not None]
    _flush_json_atomic(output_path, final_out)

    t_end = time.time()
    elapsed = t_end - t_start
    avg_gen = (total_gen_time / max(1, gen_batches))
    print(f"Saved (final): {output_path}")
    print(f"Total time (load -> final save): {elapsed:.2f} seconds")
    print(f"Avg model.generate() time per batch: {avg_gen:.4f} seconds ({gen_batches} batches)")



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="Input JSON file (list of objects).")
#     parser.add_argument("--output", required=True, help="Output JSON file.")
#     parser.add_argument(
#         "--model-name",
#         default="meta-llama/Meta-Llama-3-8B-Instruct",
#         help="HF model name/path for response generation.",
#     )

#     # which field to read as the prompt
#     parser.add_argument("--prompt-key", default="prompt_bn", help="Field name holding the prompt text.")
#     parser.add_argument("--id-key", default="id", help="Field name holding the id.")

#     # generation params
#     parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt.")
#     parser.add_argument("--max-new-tokens", type=int, default=512)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--top-p", type=float, default=0.9)
#     parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (no sampling).")

#     # behavior
#     parser.add_argument("--dedup", action="store_true", help="Deduplicate identical prompt texts.")
#     parser.add_argument("--keep-input-fields", action="store_true",
#                         help="If set, copy full original obj into output entry as `input_obj`.")

#     args = parser.parse_args()
#     print("args:", args)

#     data = load_json(args.input)
#     if not isinstance(data, list):
#         raise ValueError(f"Expected JSON root to be a list, got {type(data).__name__}")

#     gen = LLMGenerator(args.model_name)

#     seen = set()
#     out: List[Dict[str, Any]] = []

#     total = len(data)
#     processed = 0

#     for obj in data:
#         if not isinstance(obj, dict):
#             continue

#         obj_id = obj.get(args.id_key)
#         prompt = str(obj.get(args.prompt_key, "")).strip()

#         # If prompt missing, still write stub entry
#         if not prompt:
#             out_entry: Dict[str, Any] = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": "",
#                 "error": "missing_prompt",
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)
#             continue

#         h = sha1(prompt)
#         if args.dedup and h in seen:
#             # Keep an entry noting it was deduped (so ids are preserved)
#             out_entry = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": "",
#                 "deduped": True,
#                 "dedup_key": h,
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)
#             continue
#         seen.add(h)

#         try:
#             response = gen.generate_one(
#                 prompt,
#                 system=args.system,
#                 max_new_tokens=args.max_new_tokens,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 do_sample=(not args.greedy),
#             )
#             out_entry = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": response,
#                 "model": args.model_name,
#                 "gen_params": {
#                     "max_new_tokens": args.max_new_tokens,
#                     "temperature": args.temperature,
#                     "top_p": args.top_p,
#                     "greedy": bool(args.greedy),
#                 },
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)

#         except Exception as e:
#             out_entry = {
#                 args.id_key: obj_id,
#                 args.prompt_key: prompt,
#                 "response": "",
#                 "error": repr(e),
#                 "model": args.model_name,
#             }
#             if args.keep_input_fields:
#                 out_entry["input_obj"] = obj
#             out.append(out_entry)

#         processed += 1
#         percent = (processed / total) * 100
#         print(f"[Progress] {processed}/{total} ({percent:.2f}%)")

#     write_json(args.output, out)
#     print(f"Saved: {args.output}")