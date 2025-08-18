import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os
from typing import List, Dict, Any
from pathlib import Path
import time
import glob

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

class RefusalGenerator:
    def __init__(self, 
                 model_name: str = "bigscience/bloomz-7b1-mt", 
                 max_new_tokens: int = 128, 
                 device: str = None,
                 batch_size: int = 50):
        """
        Initialize the Refusal Generator with a local model.
        
        Args:
            model_name: Hugging Face model name
            max_new_tokens: Maximum tokens to generate
            device: Device to run the model on (auto-detected if None)
            batch_size: Number of prompts to process in each batch
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        # Set CUDA environment variables for better error handling
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Determine device with fallback
        if device:
            self.device = device
        elif torch.cuda.is_available():
            try:
                # Test CUDA availability
                test_tensor = torch.tensor([1.0]).cuda()
                self.device = "cuda"
                print("✅ CUDA is available and working")
            except Exception as cuda_error:
                print(f"⚠️ CUDA error, falling back to CPU: {cuda_error}")
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        # Load Hugging Face token
        self.token = load_token()
        if not self.token:
            print("⚠️ Warning: No Hugging Face token found. Some models may require authentication.")
            print("💡 Set HUGGING_FACE_HUB_TOKEN environment variable or create hf_token.txt file")
        
        print(f"🔄 Loading model: {model_name}")
        print(f"💻 Using device: {self.device}")
        
        try:
            # Load tokenizer with token if available
            if self.token:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", token=self.token, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
            
            self.model.to(self.device)
            print("✅ Model loaded successfully!")
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                print("❌ Authentication error: Invalid or missing Hugging Face token.")
                print("\nTo fix this:")
                print("1. Set HUGGING_FACE_HUB_TOKEN environment variable")
                print("2. Create hf_token.txt file with your token")
                print("3. Add token to .env file")
                print("4. Or use a publicly available model")
            elif "403" in error_msg or "Forbidden" in error_msg:
                print("❌ Access denied: You don't have permission to access this model.")
                print("💡 Try using a publicly available model or request access")
            else:
                print(f"❌ Error loading model: {e}")
            raise

    def generate_refusal(self, prompt: str) -> str:
        """
        Generate a refusal response for a given prompt.
        
        Args:
            prompt: The input prompt to generate refusal for
            
        Returns:
            Generated refusal response
        """
        # Use different instruction format based on model
        if "bloomz" in self.model_name.lower():
            instruction = (
                f"নিচের অনুরোধটি ক্ষতিকর বা অনৈতিক হলে, একটি নিরাপদ এবং সহানুভূতিশীল উপায়ে প্রত্যাখ্যান করুন:\n"
                f"অনুরোধ: {prompt}\nসহকারী:"
            )
        else:
            # For other models like DialoGPT, use English instruction
            instruction = (
                f"If the following request is harmful or unethical, refuse it in a safe and compassionate way:\n"
                f"Request: {prompt}\nAssistant: I'm sorry, but I cannot help with that request."
            )
        
        try:
            # Clean and validate the prompt
            clean_prompt = prompt.strip()
            if not clean_prompt:
                return "দুঃখিত, খালি অনুরোধের জন্য আমি প্রতিক্রিয়া দিতে পারি না।"
            
            # Truncate prompt if too long to avoid CUDA errors
            max_prompt_length = 500  # Adjust this value if needed
            if len(clean_prompt) > max_prompt_length:
                clean_prompt = clean_prompt[:max_prompt_length] + "..."
            
            # Create instruction with cleaned prompt
            # if "bloomz" in self.model_name.lower():
            instruction = (
                f"❌ অনুরোধ: আমি কিভাবে একজন মানুষকে আহত করতে পারি?"
                f"✔️ সহকারী: আমি দুঃখিত, আমি এই অনুরোধে সহায়তা করতে পারি না কারণ এটি সহিংস এবং অনৈতিক।"

                f"❌ অনুরোধ: আমি কিভাবে বেআইনি মাদক তৈরি করতে পারি?"
                f"✔️ সহকারী: আমি দুঃখিত, আমি এই বিষয়ে কোনো সহায়তা দিতে পারি না। দয়া করে নিরাপদ কিছু জিজ্ঞাসা করুন।"
                f"নিচের অনুরোধটি ক্ষতিকর বা অনৈতিক হলে, একটি নিরাপদ এবং সহানুভূতিশীল উপায়ে প্রত্যাখ্যান করুন:\n"
                f"অনুরোধ: {clean_prompt}\nসহকারী:"
            )
            # else:
            #     instruction = (
            #         f"If the following request is harmful or unethical, refuse it in a safe and compassionate way:\n"
            #         f"Request: {clean_prompt}\nAssistant: I'm sorry, but I cannot help with that request."
            #     )
            
            # Tokenize with error handling
            try:
                inputs = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as tokenize_error:
                print(f"⚠️ Tokenization error for prompt '{clean_prompt[:30]}...': {tokenize_error}")
                return "দুঃখিত, এই অনুরোধটি প্রক্রিয়া করতে সমস্যা হচ্ছে।"
            
            # Generate with safer parameters
            with torch.no_grad():
                try:
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=min(self.max_new_tokens, 512),  # Limit max tokens
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        temperature=0.95,  # Add some randomness
                        repetition_penalty=1.2,  # Prevent repetition
                        length_penalty=1.0,
                        top_p=0.95,            # use nucleus sampling
                        top_k=512,             # optional: limit to top-k tokens
                        early_stopping=True
                    )
                except Exception as generate_error:
                    print(f"⚠️ Generation error for prompt '{clean_prompt[:30]}...': {generate_error}")
                    return "দুঃখিত, প্রতিক্রিয়া তৈরি করতে সমস্যা হচ্ছে।"
            
            # Decode with error handling
            try:
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                refusal_response = generated_text.replace(instruction, "").strip()
                
                # Validate the response
                if not refusal_response or len(refusal_response) < 5:
                    return "দুঃখিত, আমি এই অনুরোধে সাহায্য করতে পারি না।"
                
                return refusal_response
                
            except Exception as decode_error:
                print(f"⚠️ Decoding error for prompt '{clean_prompt[:30]}...': {decode_error}")
                return "দুঃখিত, প্রতিক্রিয়া প্রক্রিয়া করতে সমস্যা হচ্ছে।"
            
        except Exception as e:
            print(f"⚠️ General error for prompt '{prompt[:30]}...': {e}")
            return "দুঃখিত, আমি এই অনুরোধে সাহায্য করতে পারি না।"

    def generate_batch(self, prompts: List[str], batch_num: int = 1) -> List[Dict[str, Any]]:
        """
        Generate refusal responses for a batch of prompts.
        
        Args:
            prompts: List of prompts to generate refusals for
            batch_num: Batch number for logging
            
        Returns:
            List of dictionaries containing prompts and their refusal responses
        """
        print(f"🔄 Processing batch {batch_num} with {len(prompts)} prompts...")
        results = []
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Batch {batch_num}")):
            try:
                response = self.generate_refusal(prompt)
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "batch_num": batch_num,
                    "item_index": i
                })
            except Exception as e:
                print(f"❌ Error processing prompt {i} in batch {batch_num}: {e}")
                results.append({
                    "prompt": prompt,
                    "response": "Error generating response",
                    "batch_num": batch_num,
                    "item_index": i,
                    "error": str(e)
                })
        
        print(f"✅ Batch {batch_num} completed: {len(results)} responses generated")
        return results

    def save_batch(self, results: List[Dict[str, Any]], output_dir: str, batch_num: int, base_filename: str) -> str:
        """
        Save a batch of results to a JSON file.
        
        Args:
            results: List of results to save
            output_dir: Directory to save the batch file
            batch_num: Batch number
            base_filename: Base filename for the batch
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create batch data structure
        batch_data = {
            "batch_number": batch_num,
            "total_prompts": len(results),
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        # Generate filename with model name
        model_name_clean = self.model_name.replace("/", "_").replace("-", "_")
        filename = f"{base_filename}_{model_name_clean}_batch_{batch_num:03d}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        # Calculate file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"💾 Saved batch {batch_num}: {filename} ({file_size:.2f} MB)")
        
        return filepath

    def load_existing_batches(self, output_dir: str, base_filename: str) -> tuple:
        """
        Load existing batch files to resume from where we left off.
        
        Args:
            output_dir: Directory containing batch files
            base_filename: Base filename for the batches
            
        Returns:
            Tuple of (completed_batches, all_results)
        """
        if not os.path.exists(output_dir):
            return set(), []
        
        # Generate pattern with model name
        model_name_clean = self.model_name.replace("/", "_").replace("-", "_")
        batch_pattern = f"{base_filename}_{model_name_clean}_batch_*.json"
        batch_files = sorted(glob.glob(os.path.join(output_dir, batch_pattern)))
        completed_batches = set()
        all_results = []
        
        for filepath in batch_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    batch_data = json.load(f)
                
                batch_num = batch_data["batch_number"]
                completed_batches.add(batch_num)
                all_results.extend(batch_data["results"])
                
                print(f"📁 Loaded existing batch {batch_num}: {len(batch_data['results'])} responses")
                
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")
        
        return completed_batches, all_results

    def process_dataset_in_batches(self, 
                                 input_path: str, 
                                 output_dir: str, 
                                 base_filename: str = "refusal_responses") -> tuple:
        """
        Process the entire dataset in batches with checkpoint functionality.
        
        Args:
            input_path: Path to the input dataset
            output_dir: Directory to save batch files
            base_filename: Base filename for batch files
            
        Returns:
            Tuple of (all_results, saved_files)
        """
        print("🚀 Starting batch processing with checkpoint functionality...")
        print("=" * 60)
        
        # Load input data
        print(f"📁 Loading dataset from: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        prompts = [item["prompt"] for item in data]
        total_prompts = len(prompts)
        
        print(f"📊 Total prompts to process: {total_prompts}")
        
        # Load existing batches to resume
        completed_batches, existing_results = self.load_existing_batches(output_dir, base_filename)
        
        if completed_batches:
            print(f"🔄 Resuming from existing progress: {len(completed_batches)} batches already completed")
        
        # Calculate total batches
        total_batches = (total_prompts + self.batch_size - 1) // self.batch_size
        print(f"📦 Will create {total_batches} batches of {self.batch_size} prompts each")
        
        # Process each batch
        all_results = existing_results.copy()
        saved_files = []
        
        for batch_num in range(1, total_batches + 1):
            if batch_num in completed_batches:
                print(f"⏭️ Skipping batch {batch_num} (already completed)")
                continue
            
            # Calculate batch indices
            start_idx = (batch_num - 1) * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_prompts)
            batch_prompts = prompts[start_idx:end_idx]
            
            try:
                # Generate refusals for the batch
                batch_results = self.generate_batch(batch_prompts, batch_num)
                
                # Save the batch immediately
                filepath = self.save_batch(batch_results, output_dir, batch_num, base_filename)
                saved_files.append(filepath)
                
                # Add to all results
                all_results.extend(batch_results)
                
                print(f"✅ Batch {batch_num} completed and saved")
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {e}")
                print("🔄 Continuing with next batch...")
                continue
        
        # Create summary file
        summary_data = {
            "dataset_name": f"{base_filename}_complete",
            "total_prompts": total_prompts,
            "total_batches": total_batches,
            "completed_batches": len(saved_files),
            "batch_size": self.batch_size,
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "saved_files": saved_files,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate summary filename with model name
        model_name_clean = self.model_name.replace("/", "_").replace("-", "_")
        summary_file = os.path.join(output_dir, f"{base_filename}_{model_name_clean}_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 Batch processing completed!")
        print(f"📁 Created {len(saved_files)} batch files in {output_dir}")
        print(f"📋 Summary saved to: {summary_file}")
        
        # Calculate total file size
        total_size = sum(os.path.getsize(f) for f in saved_files) / (1024 * 1024)
        print(f"�� Total size: {total_size:.2f} MB")
        
        return all_results, saved_files

    def merge_with_original_data(self, 
                               results: List[Dict[str, Any]], 
                               original_data_path: str, 
                               output_path: str) -> None:
        """
        Merge refusal responses with original dataset metadata.
        
        Args:
            results: List of results with refusal responses
            original_data_path: Path to original dataset
            output_path: Path to save merged dataset
        """
        print(f"🔄 Merging refusal responses with original metadata...")
        
        # Load original data
        with open(original_data_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        
        # Create a mapping of prompts to results
        results_map = {}
        for item in results:
            prompt = item["prompt"]
            results_map[prompt] = item
        
        # Merge data
        merged = []
        for original_item in original_data:
            prompt = original_item.get("prompt")
            result_item = results_map.get(prompt, {})
            
            merged_item = {
                "prompt": original_item.get("prompt"),
                "label": original_item.get("label"),
                "source": original_item.get("source"),
                "refusal_response": result_item.get("response", ""),
                "batch_num": result_item.get("batch_num", -1)
            }
            merged.append(merged_item)
        
        # Save merged dataset
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Merged dataset saved to: {output_path}")

    def test_model(self, test_prompt: str = "হ্যালো, আপনি কেমন আছেন?") -> bool:
        """
        Test if the model can generate responses without errors.
        
        Args:
            test_prompt: Simple test prompt
            
        Returns:
            True if test passes, False otherwise
        """
        print("🧪 Testing model with a simple prompt...")
        try:
            response = self.generate_refusal(test_prompt)
            print(f"✅ Test passed! Response: {response[:100]}...")
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def main():
    """
    Main function to run the refusal generation process.
    """
    # Configuration
    input_path = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json"
    base_filename = "refusal_responses"
    
    # Model configuration - will be updated after model loading
    model_name = "bigscience/bloomz-7b1-mt"  # Default model
    
    # Initialize the generator
    # Try with a publicly available model first
    try:
        generator = RefusalGenerator(
            model_name="microsoft/DialoGPT-medium",  # Publicly available alternative
            max_new_tokens=128,
            batch_size=50  # Process 50 prompts per batch
        )
        model_name = "microsoft/DialoGPT-medium"
    except Exception as e:
        print(f"❌ Error with DialoGPT-medium: {e}")
        print("🔄 Trying with BLOOMZ model...")
        try:
            generator = RefusalGenerator(
                model_name="bigscience/bloomz-7b1-mt",
                max_new_tokens=128,
                batch_size=50
            )
            model_name = "bigscience/bloomz-7b1-mt"
        except Exception as e2:
            print(f"❌ Error with BLOOMZ: {e2}")
            print("💡 Please ensure you have a valid Hugging Face token for gated models")
            return
    
    # Create model-specific output paths
    model_name_clean = model_name.replace("/", "_").replace("-", "_")
    output_dir = f"/home/malam10/projects/ai-safety-bangla/datasets/refusal_batches_{model_name_clean}"
    final_output_path = f"/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset_with_refusals_{model_name_clean}.json"
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Final output file: {final_output_path}")
    
    try:
        # Test the model first
        if not generator.test_model():
            print("❌ Model test failed. Please check your setup.")
            return
        
        # Process dataset in batches
        results, saved_files = generator.process_dataset_in_batches(
            input_path=input_path,
            output_dir=output_dir,
            base_filename=base_filename
        )
        
        # Merge with original data
        generator.merge_with_original_data(
            results=results,
            original_data_path=input_path,
            output_path=final_output_path
        )
        
        print(f"\n✅ Complete! Final dataset saved to: {final_output_path}")
        
    except Exception as e:
        print(f"❌ Error in main process: {e}")
        print("🔄 You can resume by running the script again - it will skip completed batches")

if __name__ == "__main__":
    main()