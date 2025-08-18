import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, precision_score, recall_score, f1_score,
    accuracy_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
import torch
import os
import glob
from pathlib import Path
import warnings
from typing import Dict, List, Any, Optional, Tuple
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_token():
    """
    Load Hugging Face token from environment variable or .env file.
    """
    # Try environment variable first
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    
    # Try .env files
    env_files = ['.env', '../.env', '../../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('HUGGING_FACE_HUB_TOKEN='):
                            return line.split('=', 1)[1].strip()
            except Exception:
                continue
    
    # Try dedicated token files
    token_files = ['hf_token.txt', '.hf_token', 'token.txt']
    for token_file in token_files:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    return f.read().strip()
            except Exception:
                continue
    
    return None

class FineTunedModelEvaluator:
    def __init__(self, 
                 original_dataset_path: str,
                 cluster_balanced_path: str,
                 random_balanced_path: str,
                 results_dir: str = "./results",
                 plots_dir: str = "/home/malam10/projects/ai-safety-bangla/3/plots"):
        """
        Initialize the fine-tuned model evaluator.
        
        Args:
            original_dataset_path: Path to original unbalanced dataset
            cluster_balanced_path: Path to cluster-balanced dataset
            random_balanced_path: Path to random-balanced dataset
            results_dir: Directory containing fine-tuned models
            plots_dir: Directory to save evaluation plots
        """
        self.original_dataset_path = original_dataset_path
        self.cluster_balanced_path = cluster_balanced_path
        self.random_balanced_path = random_balanced_path
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        
        # Load token
        self.token = load_token()
        if not self.token:
            print("❌ Hugging Face token not found!")
            print("💡 Please set HUGGING_FACE_HUB_TOKEN environment variable or create a .env file")
            raise ValueError("Hugging Face token required for model loading")
        
        # Create plots directory
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.load_datasets()
        
        # Store evaluation results
        self.evaluation_results = {}
        
    def load_datasets(self):
        """Load all datasets for evaluation"""
        print("📁 Loading datasets...")
        
        # Load original dataset
        with open(self.original_dataset_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        self.original_df = pd.DataFrame(original_data)
        
        # Load cluster-balanced dataset
        with open(self.cluster_balanced_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        self.cluster_df = pd.DataFrame(cluster_data)
        
        # Load random-balanced dataset
        with open(self.random_balanced_path, 'r', encoding='utf-8') as f:
            random_data = json.load(f)
        self.random_df = pd.DataFrame(random_data)
        
        # Convert labels to numeric
        label_mapping = {'safe': 0, 'unsafe': 1}
        for df in [self.original_df, self.cluster_df, self.random_df]:
            df['label'] = df['label'].map(label_mapping)
            df = df.dropna(subset=['prompt', 'label'])
        
        print(f"📊 Datasets loaded:")
        print(f"  Original: {len(self.original_df)} samples")
        print(f"  Cluster-balanced: {len(self.cluster_df)} samples")
        print(f"  Random-balanced: {len(self.random_df)} samples")
        
        # Print label distributions
        print(f"📈 Label distributions:")
        print(f"  Original: {self.original_df['label'].value_counts().to_dict()}")
        print(f"  Cluster-balanced: {self.cluster_df['label'].value_counts().to_dict()}")
        print(f"  Random-balanced: {self.random_df['label'].value_counts().to_dict()}")
    
    def find_finetuned_models(self) -> List[str]:
        """Find LoRA fine-tuned model only"""
        model_dirs = []
        
        # Look for LoRA model only
        lora_path = "./safety_finetuned_lora"
        
        if os.path.exists(lora_path):
            model_dirs.append(lora_path)
            print(f"✅ Found LoRA model: {lora_path}")
        else:
            print(f"❌ LoRA model not found at: {lora_path}")
            print("💡 Please ensure the LoRA model is trained and saved at './safety_finetuned_lora'")
        
        return model_dirs
    
    def load_model_and_tokenizer(self, model_path: str) -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """Load a fine-tuned model and tokenizer"""
        try:
            # Check if it's a LoRA model
            if "lora" in model_path.lower():
                # Load base model first
                base_model_name = "bigscience/bloomz-7b1-mt"
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, 
                    token=self.token,
                    trust_remote_code=True
                )
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    token=self.token,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True,
                )
                
                # Load LoRA adapters
                model = PeftModel.from_pretrained(base_model, model_path)
                print(f"✅ LoRA model loaded successfully: {model_path}")
                
            else:
                # Load regular fine-tuned model
                final_model_path = os.path.join(self.results_dir, model_path, "final_model")
                tokenizer = AutoTokenizer.from_pretrained(final_model_path)
                model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
                print(f"✅ Fine-tuned model loaded successfully: {model_path}")
            
            # Handle special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model {model_path}: {e}")
            return None, None
    
    def prepare_dataset(self, df: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
        """Prepare dataset for evaluation"""
        # Create dataset
        dataset = Dataset.from_pandas(df)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["prompt"], 
                truncation=True, 
                padding=True, 
                max_length=512
            )
        
        # Don't remove label column during tokenization
        columns_to_remove = [col for col in dataset.column_names if col != "label"]
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=columns_to_remove
        )
        
        # Ensure labels are properly formatted
        def format_labels(examples):
            return {"labels": examples["label"]}
        
        tokenized_dataset = tokenized_dataset.map(format_labels, remove_columns=["label"])
        
        return tokenized_dataset
    
    def get_predictions(self, model: torch.nn.Module, tokenizer: AutoTokenizer, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from the model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i in range(0, len(dataset), 16):  # Smaller batch size for large models
                batch = dataset[i:i+16]
                
                # Handle variable length sequences by padding
                max_length = max(len(seq) for seq in batch['input_ids'])
                
                # Pad sequences to same length
                padded_input_ids = []
                padded_attention_mask = []
                
                for input_ids, attention_mask in zip(batch['input_ids'], batch['attention_mask']):
                    # Pad input_ids
                    if len(input_ids) < max_length:
                        padding_length = max_length - len(input_ids)
                        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * padding_length)
                    else:
                        padded_input_ids.append(input_ids[:max_length])
                    
                    # Pad attention_mask
                    if len(attention_mask) < max_length:
                        padding_length = max_length - len(attention_mask)
                        padded_attention_mask.append(attention_mask + [0] * padding_length)
                    else:
                        padded_attention_mask.append(attention_mask[:max_length])
                
                # Convert to tensors
                input_ids = torch.tensor(padded_input_ids).to(device)
                attention_mask = torch.tensor(padded_attention_mask).to(device)
                
                # Get predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Handle different model types
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=1)
                    pred_labels = torch.argmax(logits, dim=1)
                    
                    predictions.extend(probs[:, 1].cpu().numpy())  # Probability of unsafe class
                    true_labels.extend(batch['labels'])
                else:
                    # For causal models, we need to extract the last token prediction
                    logits = outputs.logits
                    # This is a simplified approach - you might need to adjust based on your specific setup
                    last_token_logits = logits[:, -1, :]
                    probs = torch.softmax(last_token_logits, dim=-1)
                    # For binary classification, you might need to map specific tokens to classes
                    # This is a placeholder - adjust based on your training setup
                    predictions.extend([0.5] * len(batch['labels']))  # Placeholder
                    true_labels.extend(batch['labels'])
        
        return np.array(predictions), np.array(true_labels)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
    
    def plot_performance_comparison(self, model_name: str, results: Dict[str, Dict[str, float]]):
        """Plot performance comparison across datasets"""
        datasets = ['Original', 'Cluster-Balanced', 'Random-Balanced']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Performance Comparison - {model_name}', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            values = [results[dataset][metric] for dataset in datasets]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = axes[row, col].bar(datasets, values, color=colors, alpha=0.8)
            axes[row, col].set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_ylim(0, 1)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance comparison saved for {model_name}")
    
    def plot_roc_comparison(self, model_name: str, results: Dict[str, Dict[str, float]]):
        """Plot ROC curves comparison"""
        plt.figure(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        datasets = ['Original', 'Cluster-Balanced', 'Random-Balanced']
        
        for i, dataset in enumerate(datasets):
            fpr = results[dataset]['fpr']
            tpr = results[dataset]['tpr']
            roc_auc = results[dataset]['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{dataset} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves Comparison - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_roc_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC comparison saved for {model_name}")
    
    def plot_pr_comparison(self, model_name: str, results: Dict[str, Dict[str, float]]):
        """Plot Precision-Recall curves comparison"""
        plt.figure(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        datasets = ['Original', 'Cluster-Balanced', 'Random-Balanced']
        
        for i, dataset in enumerate(datasets):
            precision = results[dataset]['precision_curve']
            recall = results[dataset]['recall_curve']
            pr_auc = results[dataset]['pr_auc']
            
            plt.plot(recall, precision, color=colors[i], lw=2, 
                    label=f'{dataset} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves Comparison - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_pr_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PR comparison saved for {model_name}")
    
    def plot_confusion_matrices(self, model_name: str, results: Dict[str, Dict[str, float]], 
                               y_true_dict: Dict[str, np.ndarray], y_pred_dict: Dict[str, np.ndarray]):
        """Plot confusion matrices for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Confusion Matrices - {model_name}', fontsize=16, fontweight='bold')
        
        datasets = ['Original', 'Cluster-Balanced', 'Random-Balanced']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, dataset in enumerate(datasets):
            y_true = y_true_dict[dataset]
            y_pred = y_pred_dict[dataset]
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Safe', 'Unsafe'], 
                       yticklabels=['Safe', 'Unsafe'],
                       ax=axes[i])
            axes[i].set_title(f'{dataset}', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrices saved for {model_name}")
    
    def save_detailed_report(self, model_name: str, results: Dict[str, Dict[str, float]]):
        """Save detailed evaluation report"""
        report_data = {
            'model_name': model_name,
            'evaluation_timestamp': str(Path().cwd()),
            'datasets': {
                'original': {
                    'samples': len(self.original_df),
                    'label_distribution': self.original_df['label'].value_counts().to_dict()
                },
                'cluster_balanced': {
                    'samples': len(self.cluster_df),
                    'label_distribution': self.cluster_df['label'].value_counts().to_dict()
                },
                'random_balanced': {
                    'samples': len(self.random_df),
                    'label_distribution': self.random_df['label'].value_counts().to_dict()
                }
            },
            'results': results
        }
        
        # Save JSON report
        report_path = os.path.join(self.plots_dir, f'{model_name}_evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Save text report
        text_report_path = os.path.join(self.plots_dir, f'{model_name}_evaluation_report.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(f"Fine-tuned Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Evaluation Date: {str(Path().cwd())}\n\n")
            
            f.write("Dataset Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original Dataset: {len(self.original_df)} samples\n")
            f.write(f"Cluster-Balanced: {len(self.cluster_df)} samples\n")
            f.write(f"Random-Balanced: {len(self.random_df)} samples\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            for dataset, metrics in results.items():
                f.write(f"\n{dataset}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"  PR AUC: {metrics['pr_auc']:.4f}\n")
        
        print(f"✅ Detailed report saved for {model_name}")
    
    def evaluate_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single model on all datasets"""
        print(f"\n🔍 Evaluating model: {model_path}")
        print("=" * 60)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(model_path)
        if model is None or tokenizer is None:
            return None
        
        # Prepare datasets
        original_dataset = self.prepare_dataset(self.original_df, tokenizer)
        cluster_dataset = self.prepare_dataset(self.cluster_df, tokenizer)
        random_dataset = self.prepare_dataset(self.random_df, tokenizer)
        
        # Get predictions for all datasets
        print("📊 Getting predictions...")
        y_pred_orig, y_true_orig = self.get_predictions(model, tokenizer, original_dataset)
        y_pred_cluster, y_true_cluster = self.get_predictions(model, tokenizer, cluster_dataset)
        y_pred_random, y_true_random = self.get_predictions(model, tokenizer, random_dataset)
        
        # Calculate metrics
        print("📈 Calculating metrics...")
        results = {
            'Original': self.calculate_metrics(y_true_orig, y_pred_orig),
            'Cluster-Balanced': self.calculate_metrics(y_true_cluster, y_pred_cluster),
            'Random-Balanced': self.calculate_metrics(y_true_random, y_pred_random)
        }
        
        # Store predictions for confusion matrices
        y_pred_dict = {
            'Original': (y_pred_orig > 0.5).astype(int),
            'Cluster-Balanced': (y_pred_cluster > 0.5).astype(int),
            'Random-Balanced': (y_pred_random > 0.5).astype(int)
        }
        y_true_dict = {
            'Original': y_true_orig,
            'Cluster-Balanced': y_true_cluster,
            'Random-Balanced': y_true_random
        }
        
        # Generate plots
        print("📊 Generating plots...")
        self.plot_performance_comparison(model_path, results)
        self.plot_roc_comparison(model_path, results)
        self.plot_pr_comparison(model_path, results)
        self.plot_confusion_matrices(model_path, results, y_true_dict, y_pred_dict)
        
        # Save detailed report
        self.save_detailed_report(model_path, results)
        
        # Store results
        self.evaluation_results[model_path] = results
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate LoRA fine-tuned model"""
        print("🚀 Starting LoRA model evaluation...")
        
        # Find LoRA model
        model_paths = self.find_finetuned_models()
        if not model_paths:
            print("❌ LoRA model not found")
            return
        
        print(f"📁 Found LoRA model: {model_paths[0]}")
        
        # Evaluate the LoRA model
        model_path = model_paths[0]
        self.evaluate_model(model_path)
        
        print(f"\n✅ LoRA model evaluation completed!")
        print(f"📁 All plots saved to: {self.plots_dir}")
        print(f"📊 Model evaluated on 3 datasets:")
        print(f"   • Original dataset: {len(self.original_df)} samples")
        print(f"   • Cluster-balanced: {len(self.cluster_df)} samples")
        print(f"   • Random-balanced: {len(self.random_df)} samples")
    
    def create_overall_comparison(self):
        """Create overall comparison of all models"""
        print("\n📊 Creating overall model comparison...")
        
        # Prepare data for comparison
        models = list(self.evaluation_results.keys())
        datasets = ['Original', 'Cluster-Balanced', 'Random-Balanced']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            # Prepare data for this metric
            data = []
            labels = []
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for dataset in datasets:
                values = [self.evaluation_results[model][dataset][metric] for model in models]
                data.append(values)
                labels.append(dataset)
            
            # Create grouped bar chart
            x = np.arange(len(models))
            width = 0.25
            
            for j, (dataset_data, color) in enumerate(zip(data, colors)):
                axes[row, col].bar(x + j*width, dataset_data, width, label=labels[j], color=color, alpha=0.8)
            
            axes[row, col].set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_xticks(x + width)
            axes[row, col].set_xticklabels([m.split('/')[-1] for m in models], rotation=45)
            axes[row, col].legend()
            axes[row, col].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'overall_model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Overall comparison saved!")

def main():
    """Main function to run LoRA model evaluation"""
    # Configuration
    original_dataset_path = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json"
    cluster_balanced_path = "/home/malam10/projects/ai-safety-bangla/datasets/cluster_balanced_dataset.json"
    random_balanced_path = "/home/malam10/projects/ai-safety-bangla/datasets/random_balanced_dataset.json"
    results_dir = "./results"
    plots_dir = "/home/malam10/projects/ai-safety-bangla/3/plots"
    
    print("🎯 LoRA Model Evaluation")
    print("=" * 50)
    print("This script will evaluate the LoRA fine-tuned model on:")
    print("  • Original unbalanced dataset")
    print("  • Cluster-balanced dataset") 
    print("  • Random-balanced dataset")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = FineTunedModelEvaluator(
        original_dataset_path=original_dataset_path,
        cluster_balanced_path=cluster_balanced_path,
        random_balanced_path=random_balanced_path,
        results_dir=results_dir,
        plots_dir=plots_dir
    )
    
    # Run evaluation
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main() 