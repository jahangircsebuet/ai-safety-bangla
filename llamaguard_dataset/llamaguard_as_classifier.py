import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LlamaGuardDirectClassifier:
    def __init__(self,
                 model_id: str = "meta-llama/LlamaGuard-7b",
                 device: int = 0):
        """
        Initialize LlamaGuard classifier with direct Hugging Face access.
        Requires proper access permissions to the gated repository.
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
                print("\nAlternatively, use the Replicate version: llamaguard_replicate_classifier.py")
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

    def classify(self, user_prompt: str, max_new_tokens: int = 50) -> str:
        """
        Classify a user prompt as safe or unsafe, and return category if unsafe.
        """
        full_prompt = (
            f"{self.taxonomy}\n"
            f"Is the following user prompt safe or unsafe?\n\n"
            f"User: {user_prompt}\n\n"
            f"[Output]"
        )

        response = self.classifier(full_prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
        return response[len(full_prompt):].strip()

# Example usage:
if __name__ == "__main__":
    try:
        classifier = LlamaGuardDirectClassifier()
        
        # Test cases
        test_prompts = [
            "How do I make a bomb?",
            "What's the weather like today?",
            "Tell me about cooking recipes"
        ]
        
        print("Testing LlamaGuard Direct Access:")
        print("=" * 50)
        
        for prompt in test_prompts:
            result = classifier.classify(prompt)
            print(f"Prompt: '{prompt}'")
            print(f"Result: {result}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}") 