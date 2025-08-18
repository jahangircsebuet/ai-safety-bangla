import json
import os
from transformers import pipeline
import time
import math

def load_token():
    """
    Load Hugging Face token from various sources.
    """
    # Try environment variable first
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    
    # Try reading from .env file manually
    env_files = ['.env', '../.env', '../../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('HUGGING_FACE_HUB_TOKEN='):
                            token = line.split('=', 1)[1].strip()
                            if token:
                                return token
            except Exception as e:
                print(f"Warning: Could not read {env_file}: {e}")
    
    # Try reading from a token file
    token_files = ['hf_token.txt', '.hf_token', 'token.txt']
    for token_file in token_files:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                    if token:
                        return token
            except Exception as e:
                print(f"Warning: Could not read {token_file}: {e}")
    
    return None

class BanglaConversationTranslator:
    def __init__(self, model_id="Helsinki-NLP/opus-mt-en-bn"):
        """
        Initialize Bengali translator using Helsinki-NLP model.
        """
        print("Loading translation model...")
        
        # Load token
        self.token = load_token()
        
        if not self.token:
            print("❌ HUGGING_FACE_HUB_TOKEN not found!")
            print("\nPlease set it in one of these ways:")
            print("1. Environment variable: export HUGGING_FACE_HUB_TOKEN='your_token'")
            print("2. .env file: HUGGING_FACE_HUB_TOKEN=your_token")
            print("3. Token file: Create hf_token.txt with your token")
            raise ValueError("HUGGING_FACE_HUB_TOKEN not found")
        
        print(f"✅ Found HF token: {self.token[:10]}...")
        
        # List of alternative models to try
        model_options = [
            "Helsinki-NLP/opus-mt-en-bn",
            "Helsinki-NLP/opus-mt-en-bengali",
            "Helsinki-NLP/opus-mt-en-bg",
            "facebook/nllb-200-distilled-600M"  # Multilingual model
        ]
        
        for i, model_id in enumerate(model_options):
            try:
                print(f"🔄 Trying model {i+1}/{len(model_options)}: {model_id}")
                
                if "nllb" in model_id:
                    # NLLB model requires different parameters
                    self.translator = pipeline(
                        "translation", 
                        model=model_id, 
                        token=self.token,
                        src_lang="eng_Latn",
                        tgt_lang="ben_Beng",
                        device=0 if os.getenv('CUDA_VISIBLE_DEVICES') else -1
                    )
                else:
                    # Standard Helsinki-NLP models
                    self.translator = pipeline(
                        "translation", 
                        model=model_id, 
                        token=self.token,
                        device=0 if os.getenv('CUDA_VISIBLE_DEVICES') else -1
                    )
                
                print(f"✅ Translation model loaded successfully: {model_id}")
                self.model_id = model_id
                break
                
            except Exception as e:
                print(f"❌ Failed to load {model_id}: {e}")
                if i == len(model_options) - 1:  # Last model
                    print("❌ All translation models failed!")
                    raise
                continue

    def translate_text(self, text: str, max_chars: int = 512) -> str:
        """
        Translate English text to Bengali.
        """
        if not text or text.strip() == "":
            return ""
        
        try:
            # Truncate text if too long
            truncated_text = text[:max_chars]
            
            if "nllb" in self.model_id:
                # NLLB model translation
                result = self.translator(truncated_text)
            else:
                # Standard Helsinki-NLP translation
                result = self.translator(truncated_text)
            
            return result[0]["translation_text"]
        except Exception as e:
            print(f"⚠️ Translation failed for text '{text[:50]}...': {e}")
            return text  # Return original text if translation fails

    def translate_conversation(self, convo: dict) -> dict:
        """
        Translate all text fields in a conversation to Bengali.
        """
        print(f"Translating conversation {convo.get('conversation_id', 'unknown')}...")
        
        # Translate main fields
        convo["prompt_bn"] = self.translate_text(convo["prompt"])
        convo["chosen_response_bn"] = self.translate_text(convo["chosen_response"])
        convo["rejected_response_bn"] = self.translate_text(convo["rejected_response"])
        
        # Translate safety fields if they exist
        if "prompt_safety" in convo:
            convo["prompt_safety_bn"] = self.translate_text(convo["prompt_safety"])
        if "chosen_safety" in convo:
            convo["chosen_safety_bn"] = self.translate_text(convo["chosen_safety"])
        if "rejected_safety" in convo:
            convo["rejected_safety_bn"] = self.translate_text(convo["rejected_safety"])
        
        return convo

    def save_batch(self, conversations: list, batch_num: int, output_dir: str, base_filename: str):
        """
        Save a batch of translated conversations to a JSON file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create batch data structure
        batch_data = {
            "dataset_name": f"{base_filename}_batch_{batch_num:03d}",
            "batch_number": batch_num,
            "total_conversations": len(conversations),
            "source": "Anthropic/hh-rlhf",
            "description": f"Safety classification dataset batch {batch_num} for AI safety research with LlamaGuard classifications",
            "classifier_used": "LlamaGuard-7b",
            "language": "Bengali",
            "source_language": "English",
            "translation_model": self.model_id,
            "conversations": conversations
        }
        
        # Generate filename
        filename = f"{base_filename}_batch_{batch_num:03d}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        # Calculate file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"💾 Saved batch {batch_num}: {filename} ({file_size:.2f} MB)")
        
        return filepath

    def process_file(self, input_path: str, output_dir: str, base_filename: str, batch_size: int = 100, max_conversations: int = None):
        """
        Process the entire dataset file and translate to Bengali, saving in batches.
        """
        print(f"Loading dataset from {input_path}...")
        
        # Load input JSON
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = data.get("conversations", [])
        total_conversations = len(conversations)
        
        if max_conversations:
            conversations = conversations[:max_conversations]
            print(f"Processing {len(conversations)} out of {total_conversations} conversations")
        else:
            print(f"Processing all {total_conversations} conversations")

        # Calculate number of batches
        num_batches = math.ceil(len(conversations) / batch_size)
        print(f"📦 Will create {num_batches} batches of {batch_size} conversations each")

        # Process conversations in batches
        current_batch = []
        batch_num = 1
        translated_count = 0
        saved_files = []
        
        for i, convo in enumerate(conversations):
            try:
                # Translate conversation
                translated_convo = self.translate_conversation(convo)
                current_batch.append(translated_convo)
                translated_count += 1
                
                # Progress update every 10 conversations
                if (i + 1) % 10 == 0:
                    print(f"✅ Translated {i + 1}/{len(conversations)} conversations")
                
                # Save batch when it reaches the batch size
                if len(current_batch) >= batch_size:
                    filepath = self.save_batch(current_batch, batch_num, output_dir, base_filename)
                    saved_files.append(filepath)
                    current_batch = []
                    batch_num += 1
                
                # Small delay to avoid overwhelming the model
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Failed to translate conversation {i}: {e}")
                continue

        # Save remaining conversations in the last batch
        if current_batch:
            filepath = self.save_batch(current_batch, batch_num, output_dir, base_filename)
            saved_files.append(filepath)

        # Create summary file
        summary_data = {
            "dataset_name": f"{base_filename}_complete",
            "total_conversations": len(conversations),
            "translated_conversations": translated_count,
            "total_batches": len(saved_files),
            "batch_size": batch_size,
            "source": "Anthropic/hh-rlhf",
            "description": "Complete safety classification dataset for AI safety research with LlamaGuard classifications",
            "classifier_used": "LlamaGuard-7b",
            "language": "Bengali",
            "source_language": "English",
            "translation_model": self.model_id,
            "batch_files": saved_files,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = os.path.join(output_dir, f"{base_filename}_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 Translation completed!")
        print(f"✅ Successfully translated {translated_count}/{len(conversations)} conversations")
        print(f"📁 Created {len(saved_files)} batch files in {output_dir}")
        print(f"📋 Summary saved to: {summary_file}")
        
        # Calculate total file size
        total_size = sum(os.path.getsize(f) for f in saved_files) / (1024 * 1024)
        print(f"💾 Total size: {total_size:.2f} MB")

def main():
    """
    Main function to run the translation process.
    """
    try:
        # Initialize translator
        translator = BanglaConversationTranslator()
        
        # Configuration
        input_file = "llama_guard_dataset_english_with_safety.json"
        output_dir = "bangla_batches"
        base_filename = "llama_guard_dataset_bangla"
        batch_size = 100  # Save every 100 conversations
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file {input_file} not found!")
            print("Please run generate_dataset_with_classifier.py first to create the English dataset.")
            return
        
        # Process the file with batch saving
        translator.process_file(
            input_path=input_file, 
            output_dir=output_dir,
            base_filename=base_filename,
            batch_size=batch_size,
            max_conversations=None  # Set to a number to limit processing
        )
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure HUGGING_FACE_HUB_TOKEN is set")
        print("2. Check your internet connection")
        print("3. Verify the input file exists")

if __name__ == "__main__":
    main() 