#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence fork warning

import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, __version__ as TR_VERSION
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def load_token():
    """Load Hugging Face token from .env or env var."""
    for env_file in ('.env', '../.env', '../../.env'):
        if os.path.exists(env_file):
            try:
                print(f"📁 Loading token from: {env_file}")
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('HUGGING_FACE_HUB_TOKEN='):
                            tok = line.split('=', 1)[1].strip()
                            if tok:
                                print("✅ Hugging Face token loaded from .env file")
                                return tok
                            else:
                                print("⚠️ HUGGING_FACE_HUB_TOKEN found but empty in .env file")
            except Exception as e:
                print(f"❌ Error reading {env_file}: {e}")
                continue
    tok = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if tok:
        print("✅ Hugging Face token loaded from environment variable")
        return tok
    print("❌ HUGGING_FACE_HUB_TOKEN not found in .env files or environment variables")
    print("💡 Ensure: HUGGING_FACE_HUB_TOKEN=your_token_here")
    return None

class BanglaSafetyClassifierTrainer:
    def __init__(self, json_path, model_names=None, num_epochs=3):
        self.json_path = json_path
        self.num_epochs = num_epochs
        self.model_names = model_names or [
            "xlm-roberta-base",
            "bert-base-multilingual-cased",
            "distilbert-base-multilingual-cased",
        ]

        # Load JSON (accept list or {"prompts":[...]})
        print(f"📁 Loading dataset from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompts" in data:
            data = data["prompts"]

        self.df = pd.DataFrame(data)
        if 'prompt' in self.df.columns:
            self.df = self.df.rename(columns={'prompt': 'text'})

        label_map = {'safe': 0, 'unsafe': 1, '0': 0, '1': 1}
        self.df['label'] = self.df['label'].astype(str).str.lower().map(label_map)
        self.df = self.df.dropna(subset=['text', 'label'])

        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print("📈 Label distribution:")
        print(self.df['label'].value_counts())

        self.dataset = Dataset.from_pandas(self.df, preserve_index=False)

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True)  # dynamic padding via Trainer

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def _build_training_args(self, output_dir: str, epochs: int, fp16: bool) -> TrainingArguments:
        """Build TrainingArguments with backward-compatible fallback."""
        # Base args that should exist on very old versions too
        base = dict(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=epochs,
            logging_steps=50,
        )
        # Modern extras (may not exist on older versions)
        modern = dict(
            warmup_ratio=0.05,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_strategy="steps",
            report_to="none",
            fp16=fp16,
        )
        # Try modern first
        try:
            return TrainingArguments(**{**base, **modern})
        except TypeError as e:
            print(f"⚠️ Detected older transformers ({TR_VERSION}). Falling back to basic TrainingArguments. ({e})")
            # Strip unsupported keys progressively
            fallback = dict(base)
            # Add the most compatible bits if available
            try:
                return TrainingArguments(**fallback)
            except TypeError as e2:
                # Last resort: remove even more (very old versions)
                minimal = dict(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=8)
                print(f"⚠️ Using minimal TrainingArguments. ({e2})")
                return TrainingArguments(**minimal)

    def train_all(self):
        self.token = load_token()
        if not self.token:
            print("❌ Cannot proceed without Hugging Face token")
            return

        for model_name in self.model_names:
            print(f"\n🔧 Fine-tuning: {model_name}")
            print("=" * 60)
            try:
                print("📥 Loading tokenizer and model...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    if getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    problem_type="single_label_classification",
                    token=self.token,
                    trust_remote_code=True
                )

                if model.get_input_embeddings().num_embeddings != len(self.tokenizer):
                    model.resize_token_embeddings(len(self.tokenizer))

                print("🔤 Tokenizing dataset...")
                cols_to_remove = [c for c in self.dataset.column_names if c != "label"]
                tokenized = self.dataset.map(self.tokenize, batched=True, remove_columns=cols_to_remove)
                tokenized = tokenized.map(lambda ex: {"labels": ex["label"]}, remove_columns=["label"])
                tokenized = tokenized.train_test_split(test_size=0.2, seed=42)
                print(f"📊 Train samples: {len(tokenized['train'])}")
                print(f"📊 Test samples: {len(tokenized['test'])}")

                out_dir = f"./results/{model_name.replace('/', '_')}"
                os.makedirs(out_dir, exist_ok=True)
                use_fp16 = torch.cuda.is_available()

                training_args = self._build_training_args(out_dir, self.num_epochs, use_fp16)

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized["train"],
                    eval_dataset=tokenized["test"],
                    tokenizer=self.tokenizer,  # deprecation warning is harmless
                    compute_metrics=self.compute_metrics,
                )

                print(f"🚀 Starting training (Trainer will run {self.num_epochs} epoch(s))...")
                trainer.train()

                print("📊 Evaluating...")
                eval_metrics = trainer.evaluate()
                print(f"📈 Final Eval: {eval_metrics}")

                best_dir = os.path.join(out_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                trainer.save_model(best_dir)
                self.tokenizer.save_pretrained(best_dir)
                easy_dir = f"./bangla_safety_classifier_{model_name.replace('/', '_')}"
                os.makedirs(easy_dir, exist_ok=True)
                trainer.save_model(easy_dir)
                self.tokenizer.save_pretrained(easy_dir)
                print(f"✅ Best model saved to: {best_dir}")
                print(f"✅ Model also saved to: {easy_dir}")

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

if __name__ == "__main__":
    dataset_path = "datasets/bangla_safety_prompt_dataset.json"
    trainer = BanglaSafetyClassifierTrainer(
        json_path=dataset_path,
        model_names=[
            # "xlm-roberta-base",
            # "bert-base-multilingual-cased",
            "distilbert-base-multilingual-cased",
        ],
        num_epochs=3,
    )
    print("🚀 Starting training for all models...")
    trainer.train_all()
    print("✅ Training completed!")
