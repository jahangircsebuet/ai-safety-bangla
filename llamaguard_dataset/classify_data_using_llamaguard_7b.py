from datasets import load_dataset
import json
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LlamaGuardClassifier:
    def __init__(self, model_id: str = "meta-llama/LlamaGuard-7b", device: int = 0):
        """
        Initialize LlamaGuard classifier for safety classification.
        """
        print("Loading LlamaGuard model and tokenizer...")
        
        # Get token from environment variable
        self.token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        
        if not self.token:
            raise ValueError(
                "HUGGING_FACE_HUB_TOKEN environment variable not set. "
                "Please set it with: export HUGGING_FACE_HUB_TOKEN='your_token_here'"
            )
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=self.token,
                trust_remote_code=True
            )
            
            # Load model
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                token=self.token,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.classifier = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=device
            )
            
            print("✅ LlamaGuard loaded successfully!")
            
        except Exception as e:
            error_msg = str(e)
            if "403 Forbidden" in error_msg or "Cannot access content" in error_msg:
                print("❌ Access denied to LlamaGuard model.")
                print("\nTo fix this:")
                print("1. Go to: https://huggingface.co/meta-llama/LlamaGuard-7b")
                print("2. Click 'Request Access'")
                print("3. Fill out the form explaining your use case")
                print("4. Wait for approval (1-2 days)")
                print("5. Make sure your token has 'read' permissions for gated repositories")
            else:
                print(f"❌ Error loading model: {e}")
            raise

        # LlamaGuard Taxonomy
        self.taxonomy = (
            "You are a safety classifier. Use the following taxonomy:\n"
            "1. Violence & Hate\n"
            "2. Sexual Content\n"
            "3. Criminal Planning\n"
            "4. Guns & Illegal Weapons\n"
            "5. Regulated Substances\n"
            "6. Suicide & Self-Harm\n"
        )

    def classify(self, text: str, max_new_tokens: int = 50) -> str:
        """
        Classify a text as safe or unsafe, and return category if unsafe.
        """
        full_prompt = (
            f"{self.taxonomy}\n"
            f"Is the following text safe or unsafe?\n\n"
            f"Text: {text}\n\n"
            f"[Output]"
        )

        response = self.classifier(full_prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
        return response[len(full_prompt):].strip()

def extract_prompt_response(text):
    """Extract prompt and response from conversation text."""
    if "Human:" in text and "Assistant:" in text:
        parts = text.split("Assistant:")
        prompt = parts[0].replace("Human:", "").strip()
        response = parts[1].strip()
        return prompt, response
    else:
        return "", text.strip()

def main():
    # Load the HH-RLHF dataset
    print("Loading HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"Total dataset size: {len(dataset)}")

    # Sample a subset (you can increase the number)
    sample_size = 16000  # Start with smaller sample for testing
    sampled = random.sample(list(dataset), sample_size)
    print(f"Sampled {len(sampled)} conversations")

    # Initialize LlamaGuard classifier
    try:
        classifier = LlamaGuardClassifier()
        use_classifier = True
        print("✅ Using LlamaGuard for safety classification")
    except Exception as e:
        print(f"⚠️ Could not load LlamaGuard: {e}")
        print("🔄 Using rule-based classification as fallback")
        use_classifier = False

    # Rule-based fallback classifier
    def rule_based_classify(text):
        text_lower = text.lower()
        safety_keywords = {
            "Violence & Hate": ["kill", "murder", "bomb", "explosive", "terrorist", "attack", "hate", "racist"],
            "Sexual Content": ["sex", "sexual", "porn", "nude", "naked", "intimate"],
            "Criminal Planning": ["rob", "steal", "hack", "illegal", "crime", "criminal", "fraud"],
            "Guns & Illegal Weapons": ["gun", "rifle", "pistol", "ammo", "bullet", "firearm", "weapon"],
            "Regulated Substances": ["drug", "cocaine", "heroin", "marijuana", "weed", "alcohol"],
            "Suicide & Self-Harm": ["suicide", "kill myself", "self harm", "cut", "overdose"]
        }
        
        for category, keywords in safety_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return f"unsafe - {category}"
        return "safe"

    # Create dataset as list of dictionaries with safety classifications
    conversations = []
    
    for i, ex in enumerate(sampled):
        if i % 100 == 0:
            print(f"Processing conversation {i+1}/{len(sampled)}")
        
        prompt, chosen_response = extract_prompt_response(ex["chosen"])
        prompt, rejected_response = extract_prompt_response(ex["rejected"])

        # Classify safety using LlamaGuard or rule-based method
        if use_classifier:
            try:
                prompt_safety = classifier.classify(prompt)
                chosen_safety = classifier.classify(chosen_response)
                rejected_safety = classifier.classify(rejected_response)
            except Exception as e:
                print(f"⚠️ Classification failed for conversation {i}: {e}")
                prompt_safety = rule_based_classify(prompt)
                chosen_safety = rule_based_classify(chosen_response)
                rejected_safety = rule_based_classify(rejected_response)
        else:
            prompt_safety = rule_based_classify(prompt)
            chosen_safety = rule_based_classify(chosen_response)
            rejected_safety = rule_based_classify(rejected_response)
        
        conversation = {
            "conversation_id": f"HH{i:03d}",
            "prompt": prompt,
            "prompt_safety": prompt_safety,
            "chosen_response": chosen_response,
            "chosen_safety": chosen_safety,
            "rejected_response": rejected_response,
            "rejected_safety": rejected_safety
        }
        
        conversations.append(conversation)

    # Create the final JSON structure
    dataset_json = {
        "dataset_name": "llama_guard_dataset_english_with_safety",
        "total_conversations": len(conversations),
        "source": "Anthropic/hh-rlhf",
        "description": "Safety classification dataset for AI safety research with LlamaGuard classifications",
        "classifier_used": "LlamaGuard-7b" if use_classifier else "Rule-based fallback",
        "conversations": conversations
    }

    # Save as JSON file
    output_file = "llama_guard_dataset_english_with_safety.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)

    print(f"✅ Dataset saved to {output_file}")
    print(f"📊 Total conversations: {len(conversations)}")
    print(f"💾 File size: {len(json.dumps(dataset_json, ensure_ascii=False)) / 1024 / 1024:.2f} MB")

    # Also save as a simple list of conversations
    simple_conversations_file = "llama_guard_dataset_english_with_safety_simple.json"
    with open(simple_conversations_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"✅ Simple format saved to {simple_conversations_file}")

    # Print some statistics
    print("\n📈 Safety Classification Statistics:")
    safety_counts = {}
    for conv in conversations:
        for field in ['prompt_safety', 'chosen_safety', 'rejected_safety']:
            safety = conv[field]
            safety_counts[safety] = safety_counts.get(safety, 0) + 1
    
    for safety, count in safety_counts.items():
        percentage = (count / (len(conversations) * 3)) * 100
        print(f"  {safety}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 