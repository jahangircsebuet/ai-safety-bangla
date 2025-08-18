import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path

class BanglaRefusalDatasetGenerator:
    def __init__(self, input_file: str = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json", 
    output_file: str = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_refusal_dataset.json"):
        """
        Initialize the Bangla Refusal Dataset Generator.
        
        Args:
            input_file: Path to the input harmful prompts file
            output_file: Path to save the generated refusal dataset
        """
        self.input_file = input_file
        self.output_file = output_file
        self.refusal_templates = [
            "দুঃখিত, আমি এই অনুরোধে সহায়তা করতে পারি না।",
            "আমি দুঃখিত, এই অনুরোধটি আমার নীতিগত সীমার বাইরে।",
            "এই অনুরোধটি অনৈতিক বা ক্ষতিকর হতে পারে, তাই আমি এটি সম্পাদন করতে পারি না।",
            "আমি অনুগ্রহ করে জানাতে চাই যে এই প্রশ্নের উত্তর দেওয়া নিরাপদ নয়।",
            "আমি এই বিষয়ে সাহায্য করতে পারছি না কারণ এটি ক্ষতিকর অনুরোধ হতে পারে।",
            "দুঃখিত, এই ধরনের অনুরোধে আমি সহায়তা করতে পারি না।",
            "আমি এই অনুরোধটি সম্পাদন করতে পারছি না কারণ এটি নিরাপদ নয়।",
            "এই বিষয়ে আমি কোনো সহায়তা দিতে পারছি না।",
            "আমি দুঃখিত, এই অনুরোধটি আমার ক্ষমতার বাইরে।",
            "এই ধরনের অনুরোধে আমি অংশগ্রহণ করতে পারি না।",
            "আমি আন্তরিকভাবে দুঃখিত, কিন্তু আমি এই অনুরোধ পূরণ করতে পারছি না।",
            "দুঃখের সঙ্গে জানাচ্ছি, আমি এই বিষয়ে সহায়তা করতে অক্ষম।",
            "আমি ক্ষমাপ্রার্থী, কিন্তু এটি আমার নীতি-সম্মত নয়।",
            "আমার আন্তরিক দুঃখ, তবে এই অনুরোধটি আমি গ্রহণ করতে পারব না।",
            "আমি সত্যিই দুঃখিত; এটি আমার নীতিগত সীমার বাইরে।",
            "ক্ষমা করে দিন, আমি এই বিষয়ে সাহায্য করতে পারছি না।",
            "ক্ষমা প্রার্থনা করছি, আমি এই ধরনের অনুরোধে সহায়তা করতে পারি না।",
            "আমি আন্তরিকভাবে বলতে দুঃখিত, তবে এই অনুরোধে আমি অংশ নিতে পারব না।",
            "দুঃখিত, আমি এই প্রশ্ন-অনুরোধ সম্মান করতে পারছি না।",
            "আমি আন্তরিকভাবে দুঃখিত, এটি আমার নীতির বিরুদ্ধে রয়েছে।"
        ]
        
    def load_harmful_prompts(self) -> List[Dict[str, Any]]:
        """
        Load harmful prompts from the input file.
        
        Returns:
            List of dictionaries containing prompts with label and source
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If input file is not valid JSON
        """
        try:
            print(f"📁 Loading prompts from: {self.input_file}")
            with open(self.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both list format and nested format
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and "conversations" in data:
                prompts = data["conversations"]
            else:
                raise ValueError("Invalid data format. Expected list of items or dict with 'conversations' key")
            
            print(f"✅ Loaded {len(prompts)} prompts")
            return prompts
            
        except FileNotFoundError:
            print(f"❌ Error: Input file '{self.input_file}' not found")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in '{self.input_file}': {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            raise
    
    def add_custom_refusal_template(self, template: str) -> None:
        """
        Add a custom refusal template to the existing templates.
        
        Args:
            template: Custom refusal response template in Bangla
        """
        if template not in self.refusal_templates:
            self.refusal_templates.append(template)
            print(f"✅ Added custom template: {template[:50]}...")
        else:
            print(f"⚠️ Template already exists: {template[:50]}...")
    
    def get_refusal_templates(self) -> List[str]:
        """
        Get the current list of refusal templates.
        
        Returns:
            List of refusal response templates
        """
        return self.refusal_templates.copy()
    
    def generate_random_refusal(self) -> str:
        """
        Generate a random refusal response from the templates.
        
        Returns:
            Randomly selected refusal response
        """
        return random.choice(self.refusal_templates)
    
    def generate_refusal_dataset(self, prompts: List[Dict[str, Any]], use_random: bool = True) -> List[Dict[str, Any]]:
        """
        Generate refusal dataset from unsafe prompts only.
        
        Args:
            prompts: List of dictionaries containing prompts with label and source
            use_random: Whether to use random template selection (True) or round-robin (False)
            
        Returns:
            List of dictionaries containing prompts and refusal responses
        """
        print("🔄 Generating refusal responses for unsafe prompts...")
        
        refusal_dataset = []
        unsafe_count = 0
        safe_count = 0
        
        for i, item in enumerate(prompts):
            # Get prompt from the item
            prompt = item.get("prompt", "")
            if not prompt:
                print(f"⚠️ Skipping item {i}: No prompt found")
                continue
            
            # Get original label and source
            original_label = item.get("label", "unknown")
            original_source = item.get("source", "unknown")
            
            # Only process unsafe prompts
            if original_label.lower() == "unsafe":
                unsafe_count += 1
                
                # Select refusal template
                if use_random:
                    refusal_response = self.generate_random_refusal()
                else:
                    # Round-robin selection
                    template_index = unsafe_count % len(self.refusal_templates)
                    refusal_response = self.refusal_templates[template_index]
                
                # Create dataset item
                dataset_item = {
                    "id": item.get("id", f"ref_{i:04d}"),
                    "prompt": prompt,
                    "refusal_response": refusal_response,
                    "label": "safe",  # Always safe since it's a refusal response
                    "source": "template_generated",
                    "original_label": original_label,  # Keep track of original label
                    "original_source": original_source,  # Keep track of original source
                    "template_index": self.refusal_templates.index(refusal_response) if not use_random else None
                }
                
                refusal_dataset.append(dataset_item)
            else:
                safe_count += 1
        
        print(f"✅ Processed {len(prompts)} total prompts:")
        print(f"  - Unsafe prompts: {unsafe_count} (generated refusal responses)")
        print(f"  - Safe prompts: {safe_count} (skipped)")
        print(f"  - Generated {len(refusal_dataset)} refusal responses")
        
        return refusal_dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Save the generated dataset to a JSON file.
        
        Args:
            dataset: List of dataset items to save
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save dataset
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Dataset saved to: {self.output_file}")
            print(f"📊 Total items: {len(dataset)}")
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            raise
    
    def save_statistics(self, dataset: List[Dict[str, Any]], stats_file: str = None) -> None:
        """
        Save dataset statistics to a JSON file.
        
        Args:
            dataset: List of dataset items
            stats_file: Path to save statistics (if None, uses default path)
        """
        try:
            # Generate statistics
            stats = self.get_dataset_statistics(dataset)
            
            # Add metadata
            stats["metadata"] = {
                "input_file": self.input_file,
                "output_file": self.output_file,
                "generation_timestamp": str(Path().cwd()),
                "total_templates_available": len(self.refusal_templates),
                "templates_used": list(stats["template_usage"].keys())
            }
            
            # Determine stats file path
            if stats_file is None:
                stats_file = self.output_file.replace(".json", "_statistics.json")
            
            # Create output directory if it doesn't exist
            stats_dir = os.path.dirname(stats_file)
            if stats_dir:
                os.makedirs(stats_dir, exist_ok=True)
            
            # Save statistics
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"❌ Error saving statistics: {e}")
            raise
    
    def get_dataset_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the generated dataset.
        
        Args:
            dataset: List of dataset items
            
        Returns:
            Dictionary containing dataset statistics
        """
        if not dataset:
            return {"error": "Empty dataset"}
        
        # Count template usage
        template_usage = {}
        for item in dataset:
            template = item.get("refusal_response", "")
            template_usage[template] = template_usage.get(template, 0) + 1
        
        # Count original labels and sources
        original_labels = {}
        original_sources = {}
        for item in dataset:
            orig_label = item.get("original_label", "unknown")
            orig_source = item.get("original_source", "unknown")
            original_labels[orig_label] = original_labels.get(orig_label, 0) + 1
            original_sources[orig_source] = original_sources.get(orig_source, 0) + 1
        
        # Calculate prompt lengths
        prompt_lengths = [len(item.get("prompt", "")) for item in dataset]
        refusal_lengths = [len(item.get("refusal_response", "")) for item in dataset]
        
        stats = {
            "total_items": len(dataset),
            "unique_templates_used": len(template_usage),
            "template_usage": template_usage,
            "original_labels": original_labels,
            "original_sources": original_sources,
            "prompt_length_stats": {
                "min": min(prompt_lengths),
                "max": max(prompt_lengths),
                "avg": sum(prompt_lengths) / len(prompt_lengths)
            },
            "refusal_length_stats": {
                "min": min(refusal_lengths),
                "max": max(refusal_lengths),
                "avg": sum(refusal_lengths) / len(refusal_lengths)
            }
        }
        
        return stats
    
    def print_statistics(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Print dataset statistics in a formatted way.
        
        Args:
            dataset: List of dataset items
        """
        stats = self.get_dataset_statistics(dataset)
        
        print("\n📊 Dataset Statistics:")
        print("=" * 50)
        print(f"Total items: {stats['total_items']}")
        print(f"Unique templates used: {stats['unique_templates_used']}")
        
        print(f"\n📋 Original data breakdown:")
        print(f"  Original labels: {stats['original_labels']}")
        print(f"  Original sources: {stats['original_sources']}")
        
        print(f"\n📏 Prompt length statistics:")
        print(f"  Min: {stats['prompt_length_stats']['min']} characters")
        print(f"  Max: {stats['prompt_length_stats']['max']} characters")
        print(f"  Average: {stats['prompt_length_stats']['avg']:.1f} characters")
        
        print(f"\n📏 Refusal length statistics:")
        print(f"  Min: {stats['refusal_length_stats']['min']} characters")
        print(f"  Max: {stats['refusal_length_stats']['max']} characters")
        print(f"  Average: {stats['refusal_length_stats']['avg']:.1f} characters")
        
        print(f"\n🎯 Template usage:")
        for template, count in stats['template_usage'].items():
            percentage = (count / stats['total_items']) * 100
            print(f"  '{template[:30]}...': {count} times ({percentage:.1f}%)")
    
    def generate_and_save(self, use_random: bool = True, print_stats: bool = True, save_stats: bool = True) -> List[Dict[str, Any]]:
        """
        Complete pipeline: load data, generate refusals, save dataset, and optionally print/save statistics.
        
        Args:
            use_random: Whether to use random template selection
            print_stats: Whether to print dataset statistics
            save_stats: Whether to save statistics to JSON file
            
        Returns:
            Generated dataset
        """
        print("🚀 Starting refusal dataset generation...")
        print("=" * 60)
        
        # Load prompts
        prompts = self.load_harmful_prompts()
        
        # Generate refusal dataset
        dataset = self.generate_refusal_dataset(prompts, use_random=use_random)
        
        # Save dataset
        self.save_dataset(dataset)
        
        # Save statistics if requested
        if save_stats:
            self.save_statistics(dataset)
        
        # Print statistics if requested
        if print_stats:
            self.print_statistics(dataset)
        
        print("\n✅ Refusal dataset generation completed!")
        return dataset

def main():
    """Main function to run the refusal dataset generation."""
    # Configuration
    input_file = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json"
    output_file = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_refusal_dataset.json"
    
    # Initialize generator
    generator = BanglaRefusalDatasetGenerator(input_file, output_file)
    
    # Optionally add custom templates
    # generator.add_custom_refusal_template("আমার নিজস্ব প্রত্যাখ্যান বার্তা।")
    
    # Generate and save dataset
    dataset = generator.generate_and_save(use_random=True, print_stats=True, save_stats=True)
    
    print(f"\n🎉 Dataset ready for fine-tuning!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 Statistics file: {output_file.replace('.json', '_statistics.json')}")

if __name__ == "__main__":
    main()