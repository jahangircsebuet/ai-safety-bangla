import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Load model
# -----------------------------
def load_model_pipe(base_model_name, tokenizer, lora_ckpt=None, use_gpu=True):
    """Load baseline or LoRA-finetuned model pipeline"""
    if use_gpu:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype="auto").to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype="auto")

    if lora_ckpt:
        model = PeftModel.from_pretrained(model, lora_ckpt)

    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if use_gpu else -1)

# -----------------------------
# Run inference
# -----------------------------
def generate_responses(pipe, df, max_new_tokens=128, batch_size=8, n=None):
    """
    Efficient batched inference using pipeline.
    Processes the dataset without looping one prompt at a time.
    """
    subset = df if n is None else df.head(n)

    prompts = [f"[USER]: {row['prompt_bn']}\n[ASSISTANT]:" for _, row in subset.iterrows()]
    outs = pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False, batch_size=batch_size)

    results = []
    for out, (_, row) in zip(outs, subset.iterrows()):
        text = out[0]["generated_text"]
        response = text.split("[ASSISTANT]:")[-1].strip()
        results.append({
            "prompt_bn": row["prompt_bn"],
            "prompt_safety": row["prompt_safety"],
            "model_response": response
        })

    return pd.DataFrame(results)

# -----------------------------
# Save inference results
# -----------------------------
def run_and_save_inference(base_model, test_df, tokenizer, output_dir, checkpoints=None):
    os.makedirs(output_dir, exist_ok=True)
    outputs_dict = {}

    # Baseline
    baseline_pipe = load_model_pipe(base_model, tokenizer)
    baseline_outputs = generate_responses(baseline_pipe, test_df)
    baseline_path = os.path.join(output_dir, "baseline_outputs.json")
    baseline_outputs.to_json(baseline_path, orient="records", force_ascii=False, indent=2)
    outputs_dict["Baseline BLOOMZ"] = baseline_path

    # Checkpoints
    if checkpoints:
        for ckpt in checkpoints:
            ckpt_path = os.path.join(output_dir, ckpt)
            pipe = load_model_pipe(base_model, tokenizer, lora_ckpt=ckpt_path)
            ckpt_outputs = generate_responses(pipe, test_df)
            save_path = os.path.join(output_dir, f"outputs_{os.path.basename(ckpt)}.json")
            ckpt_outputs.to_json(save_path, orient="records", force_ascii=False, indent=2)
            outputs_dict[f"LoRA {os.path.basename(ckpt)}"] = save_path

    return outputs_dict
