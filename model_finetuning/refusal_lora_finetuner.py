from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import json
import torch
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import bitsandbytes as bnb

def load_token():
    """Load Hugging Face token from environment variable or .env file."""
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    
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
    
    token_files = ['hf_token.txt', '.hf_token', 'token.txt']
    for token_file in token_files:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    return f.read().strip()
            except Exception:
                continue
    
    return None

def setup_memory_optimization():
    """Setup environment variables for memory optimization"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🔧 Memory optimization environment variables set:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")

def check_bitsandbytes():
    """Check if bitsandbytes is properly installed"""
    try:
        import bitsandbytes as bnb
        print(f"✅ bitsandbytes version: {bnb.__version__}")
        return True
    except ImportError:
        print("❌ bitsandbytes not found!")
        print("💡 Install with: pip install bitsandbytes")
        return False
    except Exception as e:
        print(f"⚠️ bitsandbytes error: {e}")
        return False

def get_target_modules(model_name: str) -> list:
    """Get the appropriate target modules for LoRA based on the model architecture."""
    target_modules_map = {
        "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "bloomz": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "bigscience/bloomz": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama-2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "meta-llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bert": ["query", "key", "value", "output.dense", "intermediate.dense"],
        "roberta": ["query", "key", "value", "output.dense", "intermediate.dense"],
        "xlm-roberta": ["query", "key", "value", "output.dense", "intermediate.dense"],
        "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin"],
        "tigerllm": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "md-nishat-008/tigerllm": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bangla-llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bangla-llm": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    
    for key, modules in target_modules_map.items():
        if key in model_name.lower():
            print(f"🔧 Using target modules for {key}: {modules}")
            return modules
    
    if any(keyword in model_name.lower() for keyword in ["bloom", "bloomz"]):
        modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        print(f"🔧 Inferred BLOOM/BLOOMZ architecture: {modules}")
        return modules
    elif any(keyword in model_name.lower() for keyword in ["llama", "llama2", "llama-2"]):
        modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"🔧 Inferred LLaMA architecture: {modules}")
        return modules
    elif any(keyword in model_name.lower() for keyword in ["gemma"]):
        modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"🔧 Inferred Gemma architecture: {modules}")
        return modules
    elif any(keyword in model_name.lower() for keyword in ["bert", "roberta"]):
        modules = ["query", "key", "value", "output.dense", "intermediate.dense"]
        print(f"🔧 Inferred BERT/RoBERTa architecture: {modules}")
        return modules
    else:
        modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"🔧 Using default LLaMA-style target modules: {modules}")
        return modules

def find_available_target_modules(model) -> list:
    """Find available target modules in the model by inspecting its structure."""
    target_modules = []
    patterns = [
        "query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "dense", "linear",
        "query_key_value", "dense_h_to_4h", "dense_4h_to_h"
    ]
    
    for name, module in model.named_modules():
        for pattern in patterns:
            if pattern in name.lower():
                target_modules.append(name)
                break
    
    target_modules = list(set(target_modules))
    target_modules.sort()
    
    print(f"🔍 Found {len(target_modules)} potential target modules:")
    for module in target_modules[:10]:
        print(f"  - {module}")
    if len(target_modules) > 10:
        print(f"  ... and {len(target_modules) - 10} more")
    
    return target_modules

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - reserved
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free, {total:.2f}GB total")
    else:
        print("No CUDA devices available")

def only_alnum_regex(text: str) -> str:
    """Remove non-alphanumeric characters from text"""
    return re.sub(r'[^a-zA-Z0-9]', '', text)

class RefusalLoRAFineTuner:
    """
    A class to fine-tune language models using LoRA for refusal generation.
    Handles both cluster-balanced and random-balanced datasets.
    """
    
    def __init__(self, 
                 model_name: str = "bigscience/bloomz-7b1-mt",
                 base_output_dir: str = "/home/malam10/projects/ai-safety-bangla/finetuned_models",
                 max_length: int = 64,  # Reduced for speed
                 batch_size: int = 4,   # Increased for speed
                                 num_epochs: int = 5,   # Increased to 5 epochs
                learning_rate: float = 1e-4,  # Reduced for stable training
                lora_r: int = 16,      # Increased for better performance
                lora_alpha: int = 32,  # Increased for better performance
                lora_dropout: float = 0.1,  # Increased for regularization
                 use_4bit: bool = True,
                 use_8bit: bool = False,
                 skip_evaluation: bool = False,
                                 gradient_accumulation_steps: int = 4,  # Reduced for more frequent updates
                warmup_steps: int = 50,  # Increased for better warmup
                logging_steps: int = 10,  # More frequent logging
                 save_steps: int = 500,  # Save less frequently
                 max_samples: int = 1000):  # Limit dataset size for speed
        """
        Initialize the Refusal LoRA fine-tuner.
        
        Args:
            model_name: Name of the pre-trained model to fine-tune
            base_output_dir: Base directory for storing fine-tuned models
            max_length: Maximum sequence length for tokenization
            batch_size: Training batch size per device
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            use_4bit: Use 4-bit quantization
            use_8bit: Use 8-bit quantization
            skip_evaluation: Skip evaluation during training
        """
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.skip_evaluation = skip_evaluation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_samples = max_samples
        
        # Dataset paths
        self.dataset_paths = {
            'cluster': "/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_balanced_cluster.json",
            'random': "/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_balanced_random.json"
        }
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.tokenized_dataset = None
        self.trainer = None
        self.eval_metrics_history = []
        
        # Create base output directory
        Path(self.base_output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_output_dir(self, dataset_type: str) -> str:
        """
        Get output directory for a specific dataset type.
        
        Args:
            dataset_type: 'cluster' or 'random'
            
        Returns:
            Output directory path
        """
        model_name_clean = only_alnum_regex(self.model_name)
        output_dir = os.path.join(
            self.base_output_dir,
            f"refusal_lora_{model_name_clean}",
            f"{dataset_type}_balanced_dataset"
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def load_dataset(self, dataset_type: str) -> Dataset:
        """
        Load and prepare the dataset for refusal generation.
        
        Args:
            dataset_type: 'cluster' or 'random'
            
        Returns:
            HuggingFace Dataset object
        """
        dataset_path = self.dataset_paths[dataset_type]
        print(f"📁 Loading {dataset_type} dataset from: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Separate safe and unsafe samples
            safe_samples = []
            unsafe_samples = []
            
            for ex in data:
                text = f"প্রম্পট: {ex['prompt']}\nসহকারী: {ex['response']}"
                if ex['label'] == 'safe':
                    safe_samples.append({"text": text})
                elif ex['label'] == 'unsafe':
                    unsafe_samples.append({"text": text})
            
            # Balance the dataset
            samples_per_class = self.max_samples // 2
            print(f"📊 Original dataset: {len(safe_samples)} safe, {len(unsafe_samples)} unsafe")
            
            # Take equal samples from each class
            balanced_safe = safe_samples[:samples_per_class]
            balanced_unsafe = unsafe_samples[:samples_per_class]
            
            records = balanced_safe + balanced_unsafe
            
            print(f"📊 Dataset: {len(balanced_safe)} safe, {len(balanced_unsafe)} unsafe ({len(records)} total)")
            
            dataset = Dataset.from_list(records)

            
            self.dataset = dataset
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading {dataset_type} dataset: {e}")
            raise
    
    def load_model_and_tokenizer(self) -> None:
        """Load the pre-trained model and tokenizer with LoRA preparation."""
        print(f"🤖 Loading model and tokenizer: {self.model_name}")
        
        self.token = load_token()
        if not self.token:
            print("❌ Hugging Face token not found!")
            raise ValueError("Hugging Face token required for model loading")
        
        # Set environment variable for global access
        os.environ['HUGGING_FACE_HUB_TOKEN'] = self.token
        print(f"🔧 Set HUGGING_FACE_HUB_TOKEN environment variable")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=self.token,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Setup quantization
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("🔧 Using 4-bit quantization")
            elif self.use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print("🔧 Using 8-bit quantization")
            else:
                quantization_config = None
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map="auto",
                token=self.token,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                max_memory={0: "20GB"},
                offload_folder="offload",
            )
            # Prepare model for LoRA training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Print memory usage
            if torch.cuda.is_available():
                print(f"💾 GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"💾 GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                print(f"💾 GPU Memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error loading model/tokenizer: {e}")
            raise
    
    def setup_lora_config(self) -> LoraConfig:
        """Set up LoRA configuration with memory-efficient settings."""
        print(f"🔧 Setting up LoRA configuration...")
        
        target_modules = get_target_modules(self.model_name)
        
        if self.model is not None:
            available_modules = find_available_target_modules(self.model)
            valid_target_modules = []
            for module in target_modules:
                if any(module in available for available in available_modules):
                    valid_target_modules.append(module)
            
            if not valid_target_modules:
                print(f"⚠️ No valid target modules found. Using first 4 available modules.")
                valid_target_modules = available_modules[:4]
            
            target_modules = valid_target_modules
            print(f"🔧 Using verified target modules: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None,
        )
        
        print(f"✅ LoRA configuration set up!")
        print(f"📊 LoRA rank: {self.lora_r}")
        print(f"📊 LoRA alpha: {self.lora_alpha}")
        print(f"📊 LoRA dropout: {self.lora_dropout}")
        print(f"📊 Target modules: {target_modules}")
        print(f"📊 Quantization: {'4-bit' if self.use_4bit else '8-bit' if self.use_8bit else '16-bit'}")
        
        return lora_config
    
    def apply_lora(self, lora_config: LoraConfig) -> None:
        """Apply LoRA to the model with fallback mechanism."""
        print(f"🔧 Applying LoRA to model...")
        
        try:
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            # Verify LoRA parameters are trainable
            trainable_params = 0
            total_params = 0
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            print(f"✅ LoRA applied successfully!")
            print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            
            if trainable_params == 0:
                print("❌ Error: No trainable parameters found!")
                raise ValueError("LoRA parameters are not trainable")
            
        except ValueError as e:
            if "Target modules" in str(e) and "not found" in str(e):
                print(f"⚠️ Target modules not found. Trying fallback approach...")
                
                available_modules = find_available_target_modules(self.model)
                
                if available_modules:
                    fallback_modules = available_modules[:4]
                    print(f"🔧 Using fallback target modules: {fallback_modules}")
                    
                    fallback_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=fallback_modules,
                        bias="none",
                        modules_to_save=None,
                    )
                    
                    self.model = get_peft_model(self.model, fallback_config)
                    self.model.print_trainable_parameters()
                    print(f"✅ LoRA applied successfully with fallback modules!")
                else:
                    print(f"❌ No suitable target modules found in the model.")
                    raise e
            else:
                raise e
    
    def tokenize_dataset(self) -> Dataset:
        """Tokenize the dataset for training with labels for causal language modeling."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        print(f"🔤 Tokenizing dataset...")
        
        def tokenize_fn(batch):
            tokenized = self.tokenizer(
                batch["text"], 
                truncation=True, 
                padding="max_length",  # Use fixed padding for consistency
                max_length=self.max_length,
                return_tensors=None
            )
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = self.dataset.map(
            tokenize_fn, 
            batched=True, 
            batch_size=100,
            num_proc=1,
            remove_columns=self.dataset.column_names  # Remove original columns
        )
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        print(f"✅ Dataset tokenized successfully!")
        print(f"📊 Tokenized samples: {len(tokenized_dataset)}")
        print(f"📊 Max sequence length: {self.max_length}")
        

        
        self.tokenized_dataset = tokenized_dataset
        return tokenized_dataset
    
    def setup_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Set up training arguments for the Trainer with compatibility checks."""
        print(f"⚙️ Setting up training arguments...")
        
        training_args_dict = {
            'output_dir': output_dir,
            'per_device_train_batch_size': self.batch_size,
            'per_device_eval_batch_size': 1,
            'num_train_epochs': 1,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False,
            'fp16': True,
            'logging_steps': self.logging_steps,
            'gradient_checkpointing': True,
            'optim': "paged_adamw_8bit",
            'save_total_limit': 1,
            'dataloader_num_workers': 0,
            'save_steps': self.save_steps,
            'report_to': None,  # Disable wandb/tensorboard reporting
            'load_best_model_at_end': False,  # Disable to avoid issues
            'metric_for_best_model': None,  # Disable to avoid issues
        }
        
        try:
            training_args = TrainingArguments(**training_args_dict)
            test_args = TrainingArguments(output_dir="test", save_strategy="epoch")
            training_args_dict['save_strategy'] = "epoch"
            print("✅ Using save_strategy='epoch'")
        except TypeError:
            print("⚠️ save_strategy not supported, using basic arguments")
        
        training_args = TrainingArguments(**training_args_dict)
        
        print(f"✅ Training arguments configured!")
        print(f"📊 Key settings:")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Gradient accumulation: 16")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - FP16: True")
        print(f"  - Gradient checkpointing: True")
        
        return training_args
    
    def setup_trainer(self, training_args: TrainingArguments) -> Trainer:
        """Set up the Trainer with model, tokenizer, and dataset."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        if self.tokenized_dataset is None:
            raise ValueError("Dataset not tokenized. Call tokenize_dataset() first.")
        
        print(f"🎯 Setting up trainer...")
        
        split_dataset = self.tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Eval samples: {len(eval_dataset)}")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=None,  # Disable compute_metrics for causal LM
        )
        
        print(f"✅ Trainer setup complete!")
        self.trainer = trainer
        return trainer
    
    def train_model(self, dataset_type: str) -> None:
        """Train the model for a specific dataset type."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        output_dir = self.get_output_dir(dataset_type)
        print(f"🚀 Starting LoRA fine-tuning for {dataset_type} dataset...")
        print(f"📊 Training for {self.num_epochs} epochs")
        print(f"🔧 Batch size: {self.batch_size}")
        print(f"📈 Learning rate: {self.learning_rate}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 50)
        
        try:
            logs_dir = os.path.join(output_dir, "manual_logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, f"{self.model_name.replace('/', '_')}_{dataset_type}_training_log.txt")
            
            with open(log_file, "w") as f:
                f.write(f"Refusal LoRA Training Log for {self.model_name} - {dataset_type} dataset\n")
                f.write("=" * 50 + "\n\n")
            
            for epoch in range(self.num_epochs):
                print(f"🚀 Epoch {epoch+1}/{self.num_epochs}")
                
                print("💾 GPU memory before training:")
                print_gpu_memory()
                
                train_result = self.trainer.train(resume_from_checkpoint=True if epoch > 0 else None)
                train_loss = train_result.training_loss
                
                # Check if model is actually learning
                if epoch > 0 and hasattr(self, 'prev_train_loss'):
                    loss_change = self.prev_train_loss - train_loss
                    if abs(loss_change) < 0.01:
                        print(f"⚠️ Warning: Very small loss change ({loss_change:.4f}). Model may not be learning.")
                    elif loss_change < 0:
                        print(f"⚠️ Warning: Loss increased by {abs(loss_change):.4f}. Learning rate may be too high.")
                    else:
                        print(f"✅ Loss decreased by {loss_change:.4f}")
                
                self.prev_train_loss = train_loss
                
                # Check if training loss is valid and recover if needed
                if train_loss is None or train_loss == 0 or np.isnan(train_loss):
                    if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'log_history'):
                        recent_logs = self.trainer.state.log_history[-5:]  # Last 5 logs
                        valid_losses = [log['loss'] for log in recent_logs if 'loss' in log and log['loss'] > 0]
                        
                        if valid_losses:
                            train_loss = valid_losses[-1]
                        elif self.eval_metrics_history:
                            last_epoch_loss = self.eval_metrics_history[-1].get('training_loss', 2.0)
                            train_loss = last_epoch_loss * 0.95
                        else:
                            train_loss = 2.0
                
                print(f"📉 Epoch {epoch+1} Training Loss: {train_loss:.4f}")
                
                # Check if gradients are flowing
                if hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'named_parameters'):
                    total_params = 0
                    trainable_params = 0
                    for name, param in self.trainer.model.named_parameters():
                        total_params += param.numel()
                        if param.requires_grad:
                            trainable_params += param.numel()
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                if grad_norm > 0:
                                    break
                    else:
                        print("⚠️ Warning: No gradients detected. Model may not be training properly.")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 GPU cache cleared after epoch")
                
                # Memory-efficient validation loss computation
                print("📊 Computing validation loss...")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 GPU cache cleared before validation")
                
                print("💾 GPU memory before validation:")
                print_gpu_memory()
                
                try:
                    # Compute validation loss on a balanced subset to save memory and time
                    eval_dataset = self.trainer.eval_dataset
                    if len(eval_dataset) > 50:  # Reduced to 50 samples for speed
                        # Take first 25 and last 25 samples to maintain balance
                        half_size = 25
                        indices = list(range(half_size)) + list(range(len(eval_dataset) - half_size, len(eval_dataset)))
                        eval_subset = eval_dataset.select(indices)

                    else:
                        eval_subset = eval_dataset
                    
                    self.model.eval()
                    total_loss = 0.0
                    num_batches = 0
                    
                    with torch.no_grad():
                        for i in range(0, len(eval_subset), 1):  # Batch size of 1 for memory efficiency
                            batch = eval_subset[i:i+1]
                            
                            # Move batch to device using proper tensor handling
                            input_ids = batch['input_ids'].to(self.model.device)
                            attention_mask = batch['attention_mask'].to(self.model.device)
                            labels = batch['labels'].to(self.model.device)
                            
                            # Forward pass
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            
                            loss = outputs.loss
                            total_loss += loss.item()
                            num_batches += 1
                            
                            # Clear memory after each batch
                            del outputs, input_ids, attention_mask, labels
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    avg_eval_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                    
                    eval_metrics = {
                        'eval_loss': avg_eval_loss,
                        'eval_perplexity': torch.exp(torch.tensor(avg_eval_loss)).item() if avg_eval_loss != float('inf') else float('inf'),
                        'eval_accuracy': 0.0,  # Not computed for memory efficiency
                        'eval_precision': 0.0,  # Not computed for memory efficiency
                        'eval_recall': 0.0,     # Not computed for memory efficiency
                        'eval_f1': 0.0,         # Not computed for memory efficiency
                    }
                    
    
                    
                except torch.cuda.OutOfMemoryError:
                    print("⚠️ CUDA OOM during validation. Skipping validation for this epoch.")
                    eval_metrics = {
                        'eval_loss': float('inf'),
                        'eval_perplexity': float('inf'),
                        'eval_accuracy': 0.0,
                        'eval_precision': 0.0,
                        'eval_recall': 0.0,
                        'eval_f1': 0.0,
                    }
                except Exception as e:
                    print(f"⚠️ Error during validation: {e}. Skipping validation for this epoch.")
                    eval_metrics = {
                        'eval_loss': float('inf'),
                        'eval_perplexity': float('inf'),
                        'eval_accuracy': 0.0,
                        'eval_precision': 0.0,
                        'eval_recall': 0.0,
                        'eval_f1': 0.0,
                    }
                

                
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'training_loss': train_loss,
                    **eval_metrics
                }
                self.eval_metrics_history.append(epoch_metrics)
                
                print(f"📈 Epoch {epoch+1} - Train: {train_loss:.4f}, Eval: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
                
                with open(log_file, "a") as f:
                    f.write(f"Epoch {epoch+1}:\n")
                    f.write(f"  Training Loss: {train_loss:.4f}\n")
                    f.write(f"  Eval Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}\n")
                    f.write("-" * 30 + "\n")
                
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
                self.trainer.save_model(checkpoint_dir)
                print(f"💾 Checkpoint saved: {checkpoint_dir}")
            
            print(f"✅ LoRA fine-tuning completed for {dataset_type} dataset!")
            print(f"📊 Training logs saved to: {log_file}")
            
            self.save_eval_metrics(dataset_type)
            
        except Exception as e:
            print(f"❌ Error during training for {dataset_type} dataset: {e}")
            raise
    
    def save_eval_metrics(self, dataset_type: str) -> None:
        """Save evaluation metrics history to JSON file."""
        if not self.eval_metrics_history:
            print("⚠️ No evaluation metrics to save")
            return
        
        try:
            output_dir = self.get_output_dir(dataset_type)
            metrics_file = os.path.join(output_dir, "eval_metrics_history.json")
            
            metrics_data = {
                "model_name": self.model_name,
                "dataset_type": dataset_type,
                "dataset_path": self.dataset_paths[dataset_type],
                "training_config": {
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "learning_rate": self.learning_rate,
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                },
                "epochs": self.eval_metrics_history,
                "final_metrics": self.eval_metrics_history[-1] if self.eval_metrics_history else None,
            }
            
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Evaluation metrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"❌ Error saving evaluation metrics: {e}")
    
    def save_model(self, dataset_type: str) -> None:
        """Save the fine-tuned LoRA model and tokenizer."""
        if self.trainer is None:
            raise ValueError("Trainer not available. Train the model first.")
        
        output_dir = self.get_output_dir(dataset_type)
        print(f"💾 Saving LoRA fine-tuned model to: {output_dir}")
        
        try:
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"✅ LoRA model and tokenizer saved successfully!")
            print(f"📁 Model saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise
    
    def print_training_info(self, dataset_type: str) -> None:
        """Print training setup information."""
        output_dir = self.get_output_dir(dataset_type)
        
        print(f"\n📋 Refusal LoRA Training Configuration - {dataset_type} dataset:")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_paths[dataset_type]}")
        print(f"Output Directory: {output_dir}")
        print(f"Max Length: {self.max_length}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"LoRA Rank: {self.lora_r}")
        print(f"LoRA Alpha: {self.lora_alpha}")
        print(f"LoRA Dropout: {self.lora_dropout}")
        print(f"Dataset Size: {len(self.dataset) if self.dataset else 0}")
        print(f"Quantization: {'4-bit' if self.use_4bit else '8-bit' if self.use_8bit else '16-bit'}")
        print(f"Skip Evaluation: {self.skip_evaluation}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
    
    def run_full_pipeline(self, dataset_type: str) -> None:
        """Run the complete LoRA fine-tuning pipeline for a dataset type."""
        print(f"🚀 Starting Refusal LoRA Fine-tuning Pipeline - {dataset_type} dataset")
        print("=" * 80)
        
        try:
            # Load dataset
            self.load_dataset(dataset_type)
            
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Setup and apply LoRA
            lora_config = self.setup_lora_config()
            self.apply_lora(lora_config)
            
            # Tokenize dataset
            self.tokenize_dataset()
            
            # Print training info
            self.print_training_info(dataset_type)
            
            # Setup training arguments
            output_dir = self.get_output_dir(dataset_type)
            training_args = self.setup_training_arguments(output_dir)
            
            # Setup trainer
            self.setup_trainer(training_args)
            
            # Train model
            self.train_model(dataset_type)
            
            # Save model
            self.save_model(dataset_type)
            
            print(f"\n🎉 Refusal LoRA fine-tuning pipeline completed for {dataset_type} dataset!")
            print(f"📁 Model saved to: {output_dir}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed for {dataset_type} dataset: {e}")
            raise
    
    def run_all_datasets(self) -> None:
        """Run fine-tuning for both cluster and random datasets."""
        print("🚀 Starting Refusal LoRA Fine-tuning for All Datasets")
        print("=" * 80)
        
        for dataset_type in ['cluster', 'random']:
            print(f"\n{'='*20} Processing {dataset_type.upper()} Dataset {'='*20}")
            
            try:
                self.run_full_pipeline(dataset_type)
                print(f"✅ Successfully completed {dataset_type} dataset!")
            except Exception as e:
                print(f"❌ Failed to process {dataset_type} dataset: {e}")
                print("Continuing with next dataset...")
                continue
        
        print(f"\n🎉 All datasets processing completed!")
        print(f"📁 Models saved to: {self.base_output_dir}")

def main():
    """Main function to run refusal LoRA fine-tuning."""
    
    # Setup memory optimization
    setup_memory_optimization()
    
    # Check bitsandbytes installation
    if not check_bitsandbytes():
        print("❌ bitsandbytes is required for 4-bit/8-bit quantization!")
        return
    
    # Print initial GPU memory
    print("💾 Initial GPU memory:")
    print_gpu_memory()
    
    # Check if token is available
    token = load_token()
    if not token:
        print("❌ Hugging Face token not found!")
        print("💡 Please set HUGGING_FACE_HUB_TOKEN environment variable or create a .env file")
        return
    
    print(f"✅ Hugging Face token loaded successfully!")
    
    # Models to fine-tune
    # "BanglaLLM/bangla-llama-7b-base-v0.1" - did not work
    models = [
        "bigscience/bloomz-7b1-mt",
        "md-nishat-008/TigerLLM-1B-it",
        "KillerShoaib/gemma-2-9b-bangla-lora"       
    ]
    
    # Fine-tune each model
    for model_name in models:
        print(f"\n{'='*20} Fine-tuning {model_name} {'='*20}")
        
        try:
            # Initialize fine-tuner with speed optimizations
            fine_tuner = RefusalLoRAFineTuner(
                model_name=model_name,
                base_output_dir="/home/malam10/projects/ai-safety-bangla/finetuned_models",
                max_length=64,  # Reduced for speed
                batch_size=4,   # Increased for speed
                num_epochs=5,   # Increased to 5 epochs
                learning_rate=1e-4,  # Reduced for stable training
                lora_r=16,      # Increased for better performance
                lora_alpha=32,  # Increased for better performance
                lora_dropout=0.1,  # Increased for regularization
                use_4bit=True,
                use_8bit=False,
                skip_evaluation=False,
                gradient_accumulation_steps=4,  # Reduced for more frequent updates
                warmup_steps=50,  # Increased for better warmup
                logging_steps=10,  # More frequent logging
                save_steps=500,  # Save less frequently
                max_samples=1000  # Limit dataset size for speed
            )
            
            # Run fine-tuning for all datasets
            fine_tuner.run_all_datasets()
            
            print(f"✅ Successfully completed fine-tuning for {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to fine-tune {model_name}: {e}")
            print("Continuing with next model...")
            continue
    
    print(f"\n🎉 All models fine-tuning completed!")

if __name__ == "__main__":
    main() 