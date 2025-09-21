#!/usr/bin/env python3
"""
Fine-tune transformer models to classify Bangla Aegis responses as safe or unsafe.

Train file: datasets/converted_aegis_response_bangla.json
Test file : datasets/converted_aegis_response_bangla_TEST.json
"""

import os, json
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ====== CONFIG ======
TRAIN_FILE = "datasets/converted_aegis_response_bangla.json"
TEST_FILE  = "datasets/converted_aegis_response_bangla_TEST.json"

MODELS = [
    #"xlm-roberta-base",
    "bert-base-multilingual-cased",
    #"distilbert-base-multilingual-cased",
]
EPOCHS = 3
BATCH_SIZE = 8
FP16 = False
# ====================

def load_token():
    for env_file in (".env", "../.env", "../../.env"):
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("HUGGING_FACE_HUB_TOKEN="):
                        return line.split("=", 1)[1].strip()
    return os.getenv("HUGGING_FACE_HUB_TOKEN")


class ManualLogCallback(TrainerCallback):
    def __init__(self, json_path: str):
        self.json_path = json_path
        self._cur_epoch = None
        self._epoch_losses = []
        self._history = []
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._cur_epoch = int(state.epoch) if state.epoch is not None else None
        self._epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            try:
                self._epoch_losses.append(float(logs["loss"]))
            except Exception:
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = self._cur_epoch if self._cur_epoch is not None else int(state.epoch or 0)
        mean_train = sum(self._epoch_losses) / len(self._epoch_losses) if self._epoch_losses else None
        record = {"epoch": epoch, "train_loss": mean_train}
        if metrics:
            record.update(metrics)
        self._history.append(record)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2, ensure_ascii=False)


def _rows_to_df(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Expand nested "responses" if present
    if "responses" in df.columns:
        df = pd.json_normalize(df["responses"])

    # Check required cols
    if "response" not in df.columns:
        raise ValueError(f"'response' column missing in {path}, got: {df.columns}")

    if "label" in df.columns:
        label_col = "label"
    elif "response_label" in df.columns:
        label_col = "response_label"
    else:
        raise ValueError(f"No label column found in {path}. Available: {df.columns}")

    df = df.rename(columns={"response": "text"})
    df["label"] = df[label_col].astype(str).str.lower().map({"safe": 0, "unsafe": 1, "0": 0, "1": 1})
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    return df[["text", "label"]]


def load_local_dataset(train_path, test_path):
    return {
        "train": Dataset.from_pandas(_rows_to_df(train_path), preserve_index=False),
        "test":  Dataset.from_pandas(_rows_to_df(test_path), preserve_index=False),
    }


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    token = load_token()
    if not token:
        raise RuntimeError("HUGGING_FACE_HUB_TOKEN not found")

    datasets = load_local_dataset(TRAIN_FILE, TEST_FILE)

    for model_name in MODELS:
        print(f"\n=== Training {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if getattr(tokenizer, "eos_token", None):
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        def tok_fn(batch): 
            return tokenizer(batch["text"], truncation=True, padding=True)
        tokenized_train = datasets["train"].map(tok_fn, batched=True, remove_columns=["text"])
        tokenized_test  = datasets["test"].map(tok_fn, batched=True, remove_columns=["text"])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, problem_type="single_label_classification",
            token=token, trust_remote_code=True
        )
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

        out_dir = f"./response_classifier_bangla_{model_name.replace('/', '_')}"
        log_json = os.path.join(out_dir, "logs", f"{model_name.replace('/', '_')}_log.json")

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
            fp16=FP16,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.add_callback(ManualLogCallback(log_json))

        trainer.train()
        metrics = trainer.evaluate()
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        trainer.save_model(os.path.join(out_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(out_dir, "final_model"))
        print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
