import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, precision_score, recall_score, f1_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import torch
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self, dataset_path, results_dir="./results"):
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.load_dataset()
        
    def load_dataset(self):
        """Load the dataset for evaluation"""
        print("📁 Loading dataset...")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)
        if 'prompt' in self.df.columns:
            self.df = self.df.rename(columns={'prompt': 'text'})
        
        # Convert labels to numeric
        label_mapping = {'safe': 0, 'unsafe': 1}
        self.df['label'] = self.df['label'].map(label_mapping)
        self.df = self.df.dropna(subset=['text', 'label'])
        
        # Create dataset
        self.dataset = Dataset.from_pandas(self.df)
        
        # Split for evaluation
        self.dataset = self.dataset.train_test_split(test_size=0.2, seed=42)
        
        print(f"📊 Dataset loaded: {len(self.df)} total samples")
        print(f"📈 Label distribution: {self.df['label'].value_counts().to_dict()}")
    
    def find_trained_models(self):
        """Find all trained models in the results directory"""
        model_dirs = []
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains final_model directory (from manual training)
                final_model_path = os.path.join(item_path, "final_model")
                if os.path.exists(final_model_path) and os.path.isdir(final_model_path):
                    model_dirs.append(item)
        
        return model_dirs
    
    def load_model_and_tokenizer(self, model_dir):
        """Load a trained model and tokenizer"""
        # Load from final_model directory (from manual training)
        model_path = os.path.join(self.results_dir, model_dir, "final_model")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Handle special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            print(f"❌ Error loading model {model_dir}: {e}")
            return None, None
    
    def tokenize_data(self, tokenizer):
        """Tokenize the dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding=True, 
                max_length=512
            )
        
        # Don't remove label column during tokenization
        columns_to_remove = [col for col in self.dataset["test"].column_names if col != "label"]
        tokenized_dataset = self.dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=columns_to_remove
        )
        
        # Ensure labels are properly formatted
        def format_labels(examples):
            return {"labels": examples["label"]}
        
        tokenized_dataset = tokenized_dataset.map(format_labels, remove_columns=["label"])
        
        return tokenized_dataset
    
    def get_predictions(self, model, tokenizer, dataset):
        """Get predictions from the model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for i in range(0, len(dataset), 32):  # Batch size of 32
                batch = dataset[i:i+32]
                
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
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=1)
                pred_labels = torch.argmax(logits, dim=1)
                
                predictions.extend(probs[:, 1].cpu().numpy())  # Probability of unsafe class
                true_labels.extend(batch['labels'])
        
        return np.array(predictions), np.array(true_labels)
    
    def plot_training_curves(self, model_dir, save_path):
        """Plot training vs validation loss from manual logs"""
        try:
            # Load manual training logs
            log_file = os.path.join(self.results_dir, model_dir, "manual_logs", f"{model_dir}_training_log.txt")
            if os.path.exists(log_file):
                train_losses = []
                eval_losses = []
                epochs = []
                
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.startswith("Epoch ") and ":" in line:
                        epoch_num = int(line.split()[1].split(":")[0])
                        epochs.append(epoch_num)
                    elif "Training Loss:" in line:
                        train_loss = float(line.split("Training Loss:")[1].strip())
                        train_losses.append(train_loss)
                    elif "Eval Loss:" in line and "N/A" not in line:
                        eval_loss = float(line.split("Eval Loss:")[1].strip())
                        eval_losses.append(eval_loss)
                
                if train_losses and eval_losses:
                    plt.figure(figsize=(10, 6))
                    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
                    plt.plot(epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
                    plt.title(f'Training vs Validation Loss - {model_dir}', fontsize=14, fontweight='bold')
                    plt.xlabel('Epoch', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ Training curves saved for {model_dir}")
                else:
                    print(f"⚠️ No valid loss data found in manual logs for {model_dir}")
            else:
                print(f"⚠️ No manual training logs found for {model_dir}")
        except Exception as e:
            print(f"❌ Error plotting training curves for {model_dir}: {e}")
    
    def plot_confusion_matrix(self, y_true, y_pred, model_dir, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Safe', 'Unsafe'], 
                   yticklabels=['Safe', 'Unsafe'])
        plt.title(f'Confusion Matrix - {model_dir}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved for {model_dir}")
    
    def plot_metrics_bar(self, precision, recall, f1, model_dir, save_path):
        """Plot precision, recall, F1 bar chart"""
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [precision, recall, f1]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        plt.title(f'Model Performance Metrics - {model_dir}', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance metrics saved for {model_dir}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_dir, save_path):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_dir}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC curve saved for {model_dir}")
    
    def plot_pr_curve(self, y_true, y_pred_proba, model_dir, save_path):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_dir}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PR curve saved for {model_dir}")
    
    def save_metrics_report(self, y_true, y_pred, y_pred_proba, model_dir, save_path):
        """Save detailed metrics report"""
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC and PR AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Safe', 'Unsafe'])
        
        # Save report
        report_data = {
            'model_name': model_dir,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            },
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        with open(os.path.join(save_path, 'metrics_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save text report
        with open(os.path.join(save_path, 'metrics_report.txt'), 'w') as f:
            f.write(f"Model: {model_dir}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"PR AUC: {pr_auc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        print(f"✅ Metrics report saved for {model_dir}")
        return precision, recall, f1, roc_auc, pr_auc
    
    def evaluate_model(self, model_dir):
        """Evaluate a single model and generate all plots"""
        print(f"\n🔍 Evaluating model: {model_dir}")
        print("=" * 60)
        
        # Create evaluation directory
        eval_dir = os.path.join(self.results_dir, model_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(model_dir)
        if model is None or tokenizer is None:
            return None
        
        # Tokenize data
        tokenized_dataset = self.tokenize_data(tokenizer)
        
        # Get predictions
        y_pred_proba, y_true = self.get_predictions(model, tokenizer, tokenized_dataset["test"])
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Generate plots
        self.plot_training_curves(model_dir, eval_dir)
        self.plot_confusion_matrix(y_true, y_pred, model_dir, eval_dir)
        self.plot_roc_curve(y_true, y_pred_proba, model_dir, eval_dir)
        self.plot_pr_curve(y_true, y_pred_proba, model_dir, eval_dir)
        
        # Calculate and save metrics
        precision, recall, f1, roc_auc, pr_auc = self.save_metrics_report(
            y_true, y_pred, y_pred_proba, model_dir, eval_dir
        )
        
        # Plot performance metrics
        self.plot_metrics_bar(precision, recall, f1, model_dir, eval_dir)
        
        return {
            'model': model_dir,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("🚀 Starting evaluation of all trained models...")
        
        # Find all trained models
        model_dirs = self.find_trained_models()
        if not model_dirs:
            print("❌ No trained models found in results directory")
            return
        
        print(f"📁 Found {len(model_dirs)} trained models: {model_dirs}")
        
        # Evaluate each model
        results = []
        for model_dir in model_dirs:
            result = self.evaluate_model(model_dir)
            if result:
                results.append(result)
        
        # Create comparison summary
        if results:
            self.create_comparison_summary(results)
        
        print(f"\n✅ Evaluation completed for {len(results)} models!")
    
    def create_comparison_summary(self, results):
        """Create a comparison summary of all models"""
        print("\n📊 Creating model comparison summary...")
        
        # Create comparison directory
        comparison_dir = os.path.join(self.results_dir, "model_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Prepare data for plotting
        models = [r['model'] for r in results]
        metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            values = [r[metric] for r in results]
            
            bars = axes[row, col].bar(models, values, alpha=0.8)
            axes[row, col].set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                  f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Remove the last empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison data
        comparison_data = {
            'models': models,
            'metrics': {metric: [r[metric] for r in results] for metric in metrics}
        }
        
        with open(os.path.join(comparison_dir, 'comparison_data.json'), 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"✅ Model comparison saved to {comparison_dir}")

def main():
    """Main function to run evaluation"""
    # Configuration
    dataset_path = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json"
    results_dir = "./results"
    
    # Initialize evaluator
    evaluator = ModelEvaluator(dataset_path, results_dir)
    
    # Run evaluation
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main() 
    # Expected file structure 
    # # ./results/
    # ├── distilbert-base-multilingual-cased/
    # │   ├── final_model/                    # ✅ Evaluation looks here
    # │   │   ├── config.json
    # │   │   ├── pytorch_model.bin
    # │   │   └── tokenizer.json
    # │   ├── checkpoint-epoch-1/
    # │   ├── checkpoint-epoch-2/
    # │   ├── checkpoint-epoch-3/
    # │   └── manual_logs/                    # ✅ Training curves from here
    # │       └── distilbert-base-multilingual-cased_training_log.txt
    # └── evaluation/                         # ✅ Generated evaluation plots
    #     ├── training_curves.png
    #     ├── confusion_matrix.png
    #     ├── performance_metrics.png
    #     ├── roc_curve.png
    #     ├── pr_curve.png
    #     ├── metrics_report.json
    #     └── metrics_report.txt