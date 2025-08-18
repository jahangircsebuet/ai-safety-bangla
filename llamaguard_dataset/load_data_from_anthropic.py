from datasets import load_dataset
import json
import random

# Load the HH-RLHF dataset (https://huggingface.co/datasets/Anthropic/hh-rlhf)
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
print(f"Total dataset size: {len(dataset)}")

# Sample a subset (you can increase the number)
sampled = random.sample(list(dataset), 16000)
print(f"Sampled {len(sampled)} conversations")

# Simple keyword-based heuristic for unsafe classification (replace with classifier later)
def classify(text):
    text = text.lower()
    for keyword, category in {
        "bomb": "Violence & Hate",
        "kill": "Violence & Hate",
        "drug": "Regulated or Controlled Substances",
        "suicide": "Suicide & Self-Harm",
        "weapon": "Guns & Illegal Weapons",
        "sex": "Sexual Content"
    }.items():
        if keyword in text:
            return "unsafe", category
    return "safe", "none"

# Create dataset as list of dictionaries
conversations = []
for i, ex in enumerate(sampled):
    def extract_prompt_response(text):
        if "Human:" in text and "Assistant:" in text:
            parts = text.split("Assistant:")
            prompt = parts[0].replace("Human:", "").strip()
            response = parts[1].strip()
            return prompt, response
        else:
            return "", text.strip()
    
    prompt, chosen_response = extract_prompt_response(ex["chosen"])
    prompt, rejected_response = extract_prompt_response(ex["rejected"])

    
    conversation = {
        "conversation_id": f"HH{i:03d}",
        "prompt": prompt,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response
    }
    
    conversations.append(conversation)

# Create the final JSON structure
dataset_json = {
    "dataset_name": "llama_guard_dataset_english",
    "total_conversations": len(conversations),
    "source": "Anthropic/hh-rlhf",
    "description": "Safety classification dataset for AI safety research",
    "conversations": conversations
}

# Save as JSON file
output_file = "llama_guard_dataset_english.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_json, f, ensure_ascii=False, indent=2)

print(f"✅ Dataset saved to {output_file}")
print(f"📊 Total conversations: {len(conversations)}")
print(f"💾 File size: {len(json.dumps(dataset_json, ensure_ascii=False)) / 1024 / 1024:.2f} MB")
