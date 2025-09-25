import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import transformers
print(transformers.__file__)
print(transformers.__version__)
import json
# from inference_utils import run_inference
# from safety_metrics import compute_safety_metrics
# from plot_utils import plot_safety_comparison

# -----------------------------
# Step 1: Load dataset from JSON
# -----------------------------
df = pd.read_json("/home/malam10/projects/ai-safety-bangla/datasets/finetuning_dataset_llamaguard_bangla.json")

# -----------------------------
# Step 2: Stratified split by "prompt_safety"
# -----------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["prompt_safety"]   # preserve safe/unsafe ratio
)

# -----------------------------
# Step 3: Save splits into JSON files
# -----------------------------
train_df.to_json("datasets/finetune_banglaguard_train.json", orient="records", force_ascii=False, indent=2)
test_df.to_json("datasets/finetune_banglaguard_test.json", orient="records", force_ascii=False, indent=2)

print("Safe/Unsafe ratio in full dataset:")
print(df["prompt_safety"].value_counts(normalize=True))
print("\nSafe/Unsafe ratio in train split:")
print(train_df["prompt_safety"].value_counts(normalize=True))
print("\nSafe/Unsafe ratio in test split:")
print(test_df["prompt_safety"].value_counts(normalize=True))

# -----------------------------
# Step 4: Choose model
# -----------------------------
MODEL_NAME = "bigscience/bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

# -----------------------------
# Step 5: Configure LoRA
# -----------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# Step 6: Load dataset splits
# -----------------------------
dataset = load_dataset("json", data_files={
    "train": "/home/malam10/projects/ai-safety-bangla/datasets/finetune_banglaguard_train.json",
    "test": "/home/malam10/projects/ai-safety-bangla/datasets/finetune_banglaguard_test.json"
})

# Formatting: {prompt, response} → conversation format
def format_examples(example):
    user = "[USER]: " + example["prompt"]
    assistant = "[ASSISTANT]: " + example["response"]
    text = user + "\n" + assistant
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(format_examples, batched=False)

# -----------------------------
# Step 7: Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora-bangla-llm/llamaguard-dataset/bloomz-7b1-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    # evaluation_strategy="epoch",   # auto-evaluate every epoch
    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -----------------------------
# Baseline evaluation (before training)
# -----------------------------
baseline_results = trainer.evaluate(tokenized_datasets["test"])
print("📊 Baseline evaluation results (before training):", baseline_results)

with open("eval_results_baseline_llamaguard_dataset_bloomz_7b1.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)

# -----------------------------
# Baseline evaluation - compute safety metrics -----> DO IT SEPARATELY
# -----------------------------
# baseline_results_df = run_inference(model, tokenizer, tokenized_datasets["test"])
# baseline_metrics = compute_safety_metrics(baseline_results_df,
#                                           prompt_col="prompt_bn",
#                                           response_col="model_response",
#                                           safety_col="prompt_safety")

# print("📊 Baseline Safety Metrics:", baseline_metrics)
# baseline_results_df.to_json("baseline_outputs.json", orient="records", force_ascii=False, indent=2)
# -----------------------------
# Step 10: Train
# -----------------------------
trainer.train()


# -----------------------------
# Step 11: Evaluate model (after finetuning)
# -----------------------------
import json

# Evaluate on test set after initial fine-tuning
eval_results = trainer.evaluate(tokenized_datasets["test"])
print("📊 Evaluation results (initial fine-tuning):", eval_results)

# Save results to JSON
with open("eval_results_finetuned_llamaguard_dataset_bloomz_7b1.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)


# -----------------------------
# Evaluate model (after finetuning) - compute safety metrics ---> DO IT SEPARATELY
# -----------------------------
# finetuned_results_df = run_inference(model, tokenizer, tokenized_datasets["test"])
# finetuned_metrics = compute_safety_metrics(finetuned_results_df,
#                                            prompt_col="prompt_bn",
#                                            response_col="model_response",
#                                            safety_col="prompt_safety")

# print("📊 Fine-tuned Safety Metrics:", finetuned_metrics)
# finetuned_results_df.to_json("finetuned_outputs.json", orient="records", force_ascii=False, indent=2)

# -----------------------------
# Step 12: Save LoRA adapter
# -----------------------------
model.save_pretrained("./lora-bangla-llm/llamaguard-dataset/bloomz-7b1-finetuned")
tokenizer.save_pretrained("./lora-bangla-llm/llamaguard-dataset/bloomz-7b1-finetuned")
print("✅ LoRA fine-tuned model and tokenizer saved to ./lora-bangla-llm/llamaguard-dataset/bloomz-7b1-finetuned")


# -----------------------------
# Step 12: plot safety metrics comparison - DO IT SEPARATELY
# -----------------------------
# After computing metrics:
# plot_safety_comparison(baseline_metrics, finetuned_metrics, save_path="safety_metrics_comparison.png")



# train test split and ratio category wise 
# Safe/Unsafe ratio in full dataset:
# prompt_safety
# safe                               0.564427
# unsafe - Criminal Planning         0.163687
# unsafe - Violence & Hate           0.113132
# unsafe - Regulated Substances      0.057953
# unsafe - Sexual Content            0.034525
# unsafe - Suicide & Self-Harm       0.033600
# unsafe - Guns & Illegal Weapons    0.032676
# Name: proportion, dtype: float64

# Safe/Unsafe ratio in train split:
# prompt_safety
# safe                               0.564547
# unsafe - Criminal Planning         0.163776
# unsafe - Violence & Hate           0.112909
# unsafe - Regulated Substances      0.057803
# unsafe - Sexual Content            0.034682
# unsafe - Suicide & Self-Harm       0.033526
# unsafe - Guns & Illegal Weapons    0.032755
# Name: proportion, dtype: float64

# Safe/Unsafe ratio in test split:
# prompt_safety
# safe                               0.563945
# unsafe - Criminal Planning         0.163328
# unsafe - Violence & Hate           0.114022
# unsafe - Regulated Substances      0.058552
# unsafe - Suicide & Self-Harm       0.033898
# unsafe - Sexual Content            0.033898
# unsafe - Guns & Illegal Weapons    0.032357
# Name: proportion, dtype: float64