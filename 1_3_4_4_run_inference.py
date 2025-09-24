from transformers import AutoTokenizer
import pandas as pd
from 1_3_4_2_inference_utils import run_and_save_inference

BASE_MODEL = "bigscience/bloomz-7b1"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

test_df = pd.read_json("/home/malam10/projects/ai-safety-bangla/banglaguard_test.json")

outputs_dict = run_and_save_inference(
    BASE_MODEL,
    test_df,
    tokenizer,
    output_dir="./inference_outputs",
    checkpoints=[
        "/home/malam10/projects/ai-safety-bangla/lora-bangla-llm/checkpoint-163",
        "/home/malam10/projects/ai-safety-bangla/lora-bangla-llm/checkpoint-326",
        "/home/malam10/projects/ai-safety-bangla/lora-bangla-llm/checkpoint-489"
    ]
)

import json

# Save the dictionary to a JSON file
with open("outputs_dict.json", "w", encoding="utf-8") as f:
    json.dump(outputs_dict, f, ensure_ascii=False, indent=2)

print("📂 Saved outputs_dict mapping to outputs_dict.json")


# plot graphs 
from 1_3_4_3_metrics_eval import evaluate_outputs

metrics_dict = evaluate_outputs(outputs_dict, save_plot="safety_comparison_checkpoints.png")

