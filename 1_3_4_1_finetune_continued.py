import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel

# -----------------------------
# Step 1: Load NEW dataset
# -----------------------------
df_new = pd.read_json("new_data.json")

# Stratified split (if "prompt_safety" exists in your data)
if "prompt_safety" in df_new.columns:
    train_df, test_df = train_test_split(
        df_new,
        test_size=0.2,
        random_state=42,
        stratify=df_new["prompt_safety"]
    )
else:
    train_df, test_df = train_test_split(
        df_new,
        test_size=0.2,
        random_state=42
    )

# Save splits
train_df.to_json("new_train.json", orient="records", force_ascii=False, indent=2)
test_df.to_json("new_test.json", orient="records", force_ascii=False, indent=2)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# -----------------------------
# Step 2: Load base + LoRA adapter
# -----------------------------
MODEL_NAME = "bigscience/bloomz-7b1"  # or TituLLM/TigerLLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

# Load previously fine-tuned LoRA adapter
model = PeftModel.from_pretrained(model, "./lora-bangla-llm")

# -----------------------------
# Step 3: Load dataset splits with HuggingFace
# -----------------------------
dataset = load_dataset("json", data_files={
    "train": "new_train.json",
    "test": "new_test.json"
})

def format_examples(example):
    user = "[USER]: " + example["prompt_bn"]
    assistant = "[ASSISTANT]: " + example["chosen_response_bn"]
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
# Step 4: Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# Step 5: Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora-bangla-llm-continued",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,    # lower LR for continued training
    num_train_epochs=2,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none"
)

# -----------------------------
# Step 6: Trainer
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
# Step 7: Continue training
# -----------------------------
trainer.train()


# -----------------------------
# Step 8: Evaluate model
# -----------------------------
import json

# Evaluate on test set after initial fine-tuning
eval_results = trainer.evaluate(tokenized_datasets["test"])
print("📊 Evaluation results (initial fine-tuning):", eval_results)

# Save results to JSON
with open("eval_results_initial.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

# -----------------------------
# Step 9: Save updated adapter
# -----------------------------
model.save_pretrained("./lora-bangla-llm-continued")
tokenizer.save_pretrained("./lora-bangla-llm-continued")
