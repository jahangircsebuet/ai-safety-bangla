import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os

def load_token():
    """
    Load Hugging Face token from .env file or environment variable.
    """
    # First try to load from .env file
    env_files = ['.env', '../.env', '../../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                print(f"📁 Loading token from: {env_file}")
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('HUGGING_FACE_HUB_TOKEN='):
                            token = line.split('=', 1)[1].strip()
                            if token:
                                print("✅ Hugging Face token loaded from .env file")
                                return token
                            else:
                                print("⚠️ HUGGING_FACE_HUB_TOKEN found but empty in .env file")
            except Exception as e:
                print(f"❌ Error reading {env_file}: {e}")
                continue
    
    # Fallback to environment variable
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        print("✅ Hugging Face token loaded from environment variable")
        return token
    
    print("❌ HUGGING_FACE_HUB_TOKEN not found in .env files or environment variables")
    print("💡 Make sure your .env file contains: HUGGING_FACE_HUB_TOKEN=your_token_here")
    return None

class BanglaSafetyClassifierTrainer:
    def __init__(self, json_path, model_names=None, num_epochs=3):
        self.json_path = json_path
        self.num_epochs = num_epochs
        self.model_names = model_names or [
            "xlm-roberta-base",
            "bert-base-multilingual-cased",
            "distilbert-base-multilingual-cased"  # Alternative to ai4bharat/indic-bert
        ]
        
        # Load JSON data and convert to DataFrame
        print(f"📁 Loading dataset from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame with proper column names
        self.df = pd.DataFrame(data)
        
        # Rename 'prompt' to 'text' for the tokenizer
        if 'prompt' in self.df.columns:
            self.df = self.df.rename(columns={'prompt': 'text'})
        
        # Convert labels to numeric (safe=0, unsafe=1)
        label_mapping = {'safe': 0, 'unsafe': 1}
        self.df['label'] = self.df['label'].map(label_mapping)
        
        # Remove rows with missing data
        self.df = self.df.dropna(subset=['text', 'label'])
        
        print(f"📊 Dataset loaded: {len(self.df)} samples")
        print(f"📈 Label distribution:")
        print(self.df['label'].value_counts())
        
        self.dataset = Dataset.from_pandas(self.df)

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], truncation=True, padding=True)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def train_all(self):
        # Load Hugging Face token
        self.token = load_token()
        if not self.token:
            print("❌ Cannot proceed without Hugging Face token")
            return
        
        for model_name in self.model_names:
            print(f"\n🔧 Fine-tuning: {model_name}")
            print("=" * 60)
            
            try:
                # Load tokenizer and model with token
                print(f"📥 Loading tokenizer and model...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, trust_remote_code=True)
                
                # Handle special tokens for some models
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=2,
                    problem_type="single_label_classification",
                    token=self.token,
                    trust_remote_code=True
                )

                # Tokenize dataset
                print(f"🔤 Tokenizing dataset...")
                # Don't remove label column during tokenization
                columns_to_remove = [col for col in self.dataset.column_names if col != "label"]
                tokenized_dataset = self.dataset.map(self.tokenize, batched=True, remove_columns=columns_to_remove)
                
                # Ensure labels are properly formatted for the Trainer
                def format_labels(examples):
                    return {"labels": examples["label"]}
                
                tokenized_dataset = tokenized_dataset.map(format_labels, remove_columns=["label"])
                tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

                print(f"📊 Train samples: {len(tokenized_dataset['train'])}")
                print(f"📊 Test samples: {len(tokenized_dataset['test'])}")

                # Create output directory
                output_dir = f"./results/{model_name.replace('/', '_')}"
                os.makedirs(output_dir, exist_ok=True)

                training_args_dict = {
                    "output_dir": output_dir,
                    "per_device_train_batch_size": 8,
                    "per_device_eval_batch_size": 8,
                    "num_train_epochs": 1,  # Set to 1 since we're handling epochs manually
                    "learning_rate": 2e-5,
                    "warmup_steps": 100,
                    "weight_decay": 0.01,
                    "logging_steps": 50
                }
                
                training_args = TrainingArguments(**training_args_dict)

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["test"],
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics
                )

                print(f"🚀 Starting training...")

                # Create logs directory
                logs_dir = os.path.join(output_dir, "manual_logs")
                os.makedirs(logs_dir, exist_ok=True)
                log_file = os.path.join(logs_dir, f"{model_name.replace('/', '_')}_training_log.txt")
                
                # Initialize log file
                with open(log_file, "w") as f:
                    f.write(f"Training Log for {model_name}\n")
                    f.write("=" * 50 + "\n\n")

                for epoch in range(self.num_epochs):
                    print(f"🚀 Epoch {epoch+1}/{self.num_epochs}")
                    
                    # Train for one epoch
                    train_result = trainer.train(resume_from_checkpoint=True if epoch > 0 else None)
                    
                    # Get training loss
                    train_loss = train_result.training_loss
                    print(f"📉 Epoch {epoch+1} Training Loss: {train_loss:.4f}")
                    
                    # Evaluate
                    print("📊 Evaluating...")
                    eval_metrics = trainer.evaluate()
                    eval_loss = eval_metrics.get('eval_loss', 'N/A')
                    
                    # Log metrics
                    print(f"📈 Epoch {epoch+1} Eval Loss: {eval_loss}")
                    print(f"📈 Epoch {epoch+1} Eval Metrics: {eval_metrics}")
                    
                    # Save to log file
                    with open(log_file, "a") as f:
                        f.write(f"Epoch {epoch+1}:\n")
                        f.write(f"  Training Loss: {train_loss:.4f}\n")
                        f.write(f"  Eval Loss: {eval_loss}\n")
                        f.write(f"  Eval Metrics: {eval_metrics}\n")
                        f.write("-" * 30 + "\n")
                    
                    # Save checkpoint after each epoch
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
                    trainer.save_model(checkpoint_dir)
                    print(f"💾 Checkpoint saved: {checkpoint_dir}")

                print(f"✅ Training completed! Logs saved to: {log_file}")
                
                # Evaluate final model
                print(f"📊 Evaluating final model...")
                eval_results = trainer.evaluate()
                print(f"Final metrics: {eval_results}")
                
                # Save model in results directory for evaluation
                model_save_path = os.path.join(output_dir, "final_model")
                trainer.save_model(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)
                
                # Also save in a separate directory for easy access
                easy_access_path = f"./bangla_safety_classifier_{model_name.replace('/', '_')}"
                trainer.save_model(easy_access_path)
                self.tokenizer.save_pretrained(easy_access_path)
                
                print(f"✅ Model saved to: {model_save_path}")
                print(f"✅ Model also saved to: {easy_access_path}")
                
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg:
                    print(f"❌ Authentication error for {model_name}: Invalid or missing Hugging Face token.")
                    print("💡 Please check your .env file or HUGGING_FACE_HUB_TOKEN environment variable")
                elif "403" in error_msg or "Forbidden" in error_msg:
                    print(f"❌ Access denied to {model_name}: You don't have permission to access this model.")
                    print("💡 Try using a publicly available model or request access")
                else:
                    print(f"❌ Error training {model_name}: {e}")
                continue

if __name__ == "__main__":
    # Dataset path
    dataset_path = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json"
    
    # Initialize trainer
    trainer = BanglaSafetyClassifierTrainer(
        json_path=dataset_path,
        model_names=[
            # "xlm-roberta-base",
            # "bert-base-multilingual-cased",
            "distilbert-base-multilingual-cased"  # Alternative to ai4bharat/indic-bert
        ],
        num_epochs=3
    )
    
    # Train all models
    print("🚀 Starting training for all models...")
    trainer.train_all()
    print("✅ Training completed!")
