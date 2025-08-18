import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig

# --------- Config ----------
MODEL_NAME = "bigscience/bloomz-7b1-mt"
DATA_PATH = "/path/to/your/bangla_prompt_response_balanced_cluster.json"
OUTPUT_DIR = "./bloomz_lora_finetuned"
NUM_EPOCHS = 3
MAX_LENGTH = 128
BATCH_SIZE = 1
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
USE_4BIT = True

# --------- Load Data ---------
from datasets import Dataset
import json

def load_bangla_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    return Dataset.from_list([{"text": f"প্রম্পট: {x['prompt']}\nসহকারী: {x['response']}"} for x in raw_data])

dataset = load_bangla_dataset(DATA_PATH)

# --------- Tokenization ---------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Important for BLOOM

def tokenize(sample):
    tokens = tokenizer(
        sample["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])

# --------- Model Loading (4-bit) ---------
quant_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quant_config
)
model = prepare_model_for_kbit_training(model)

# --------- Apply LoRA ---------
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------- Trainer Setup ---------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=8,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# --------- Train ---------
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
