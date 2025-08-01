import json
import glob
from pathlib import Path
from typing import List, Dict, Any

class BanglaSafetyDatasetGenerator:
    def __init__(self, batch_dir: str = "/home/malam10/projects/ai-safety-bangla/llamaguard_dataset/bangla_batches"):
        """
        Initialize the Bangla Safety Dataset Generator.
        
        Args:
            batch_dir: Directory containing the batch files
        """
        self.batch_dir = batch_dir
        self.compiled_data = []
        
    def load_batch_files(self) -> List[str]:
        """
        Load and sort batch files from the specified directory.
        
        Returns:
            List of sorted batch file paths
        """
        batch_files = sorted(glob.glob(f"{self.batch_dir}/llama_guard_dataset_bangla_batch_*.json"))
        print(f"📁 Found {len(batch_files)} batch files in {self.batch_dir}")
        return batch_files
    
    def extract_llamaguard_data(self, batch_files: List[str]) -> None:
        """
        Extract prompt_bn and prompt_safety_bn from batch files.
        
        Args:
            batch_files: List of batch file paths
        """
        print("🔄 Extracting LlamaGuard data from batch files...")
        
        for i, file in enumerate(batch_files):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    batch = json.load(f)
                    
                conversations = batch.get("conversations", [])
                extracted_count = 0
                
                for item in conversations:
                    prompt = item.get("prompt_bn", "").strip()
                    label = item.get("prompt_safety", "").strip()
                    
                    if prompt and label:
                        self.compiled_data.append({
                            "prompt": prompt,
                            "label": "safe" if label == "safe" else "unsafe",
                            "source": "llamaguard"
                        })
                        extracted_count += 1
                
                print(f"  ✅ Batch {i+1}: Extracted {extracted_count} conversations")
                
            except Exception as e:
                print(f"  ❌ Error processing {file}: {e}")
                continue
        
        print(f"📊 Total LlamaGuard data extracted: {len([x for x in self.compiled_data if x['source'] == 'llamaguard'])}")
    
    def add_multijail_data(self, file_path: str = "/home/malam10/projects/ai-safety-bangla/datasets/converted_multijail_bangla.json") -> None:
        """
        Add MultiJail prompts from the specified file.
        
        Args:
            file_path: Path to the MultiJail dataset file
        """
        print(f"🔄 Adding MultiJail data from: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                multi_data = json.load(f)
            
            added_count = 0
            prompts = multi_data.get("prompts", [])
            
            for item in prompts:
                prompt = item.get("prompt", "").strip()
                if prompt:
                    self.compiled_data.append({
                        "prompt": prompt,
                        "label": "unsafe",  # All MultiJail prompts are unsafe
                        "source": "multijail"
                    })
                    added_count += 1
            
            print(f"  ✅ Added {added_count} MultiJail prompts")
            
        except FileNotFoundError:
            print(f"  ❌ MultiJail file not found: {file_path}")
        except Exception as e:
            print(f"  ❌ Error loading MultiJail data: {e}")
    
    def add_catqa_data(self, file_path: str = "/home/malam10/projects/ai-safety-bangla/datasets/converted_catqa_bangla.json") -> None:
        """
        Add CatQA prompts from the specified file.
        
        Args:
            file_path: Path to the CatQA dataset file
        """
        print(f"🔄 Adding CatQA data from: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                catqa_data = json.load(f)
            
            added_count = 0
            prompts = catqa_data.get("prompts", [])
            
            for item in prompts:
                prompt = item.get("prompt", "").strip()
                if prompt:
                    self.compiled_data.append({
                        "prompt": prompt,
                        "label": "unsafe",  # All CatQA prompts are unsafe
                        "source": "catqa"
                    })
                    added_count += 1
            
            print(f"  ✅ Added {added_count} CatQA prompts")
            
        except FileNotFoundError:
            print(f"  ❌ CatQA file not found: {file_path}")
        except Exception as e:
            print(f"  ❌ Error loading CatQA data: {e}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the compiled dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        total_prompts = len(self.compiled_data)
        safe_prompts = len([x for x in self.compiled_data if x["label"] == "safe"])
        unsafe_prompts = len([x for x in self.compiled_data if x["label"] == "unsafe"])
        
        source_counts = {}
        for item in self.compiled_data:
            source = item.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_prompts": total_prompts,
            "safe_prompts": safe_prompts,
            "unsafe_prompts": unsafe_prompts,
            "safe_percentage": (safe_prompts / total_prompts * 100) if total_prompts > 0 else 0,
            "unsafe_percentage": (unsafe_prompts / total_prompts * 100) if total_prompts > 0 else 0,
            "source_distribution": source_counts
        }
    
    def save_dataset(self, output_path: str = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json") -> None:
        """
        Save the combined dataset to a JSON file.
        
        Args:
            output_path: Path where the dataset should be saved
        """
        print(f"💾 Saving combined dataset to: {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the dataset
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.compiled_data, f, ensure_ascii=False, indent=2)
        
        # Calculate file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✅ Dataset saved successfully! File size: {file_size:.2f} MB")
    
    def generate_dataset(self, 
                        output_path: str = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_safety_prompt_dataset.json",
                        multijail_path: str = "/home/malam10/projects/ai-safety-bangla/datasets/converted_multijail_bangla.json",
                        catqa_path: str = "/home/malam10/projects/ai-safety-bangla/datasets/converted_catqa_bangla.json") -> Dict[str, Any]:
        """
        Generate the complete Bangla safety dataset.
        
        Args:
            output_path: Path where the dataset should be saved
            multijail_path: Path to the MultiJail dataset file
            catqa_path: Path to the CatQA dataset file
            
        Returns:
            Dictionary containing dataset statistics
        """
        print("🚀 Starting Bangla Safety Dataset Generation...")
        print("=" * 60)
        
        # Step 1: Load batch files
        batch_files = self.load_batch_files()
        
        if not batch_files:
            print("❌ No batch files found!")
            return {}
        
        # Step 2: Extract LlamaGuard data
        self.extract_llamaguard_data(batch_files)
        
        # Step 3: Add MultiJail data
        self.add_multijail_data(multijail_path)
        
        # Step 4: Add CatQA data
        self.add_catqa_data(catqa_path)
        
        # Step 5: Get statistics
        stats = self.get_dataset_statistics()
        
        # Step 6: Save dataset
        self.save_dataset(output_path)
        
        # Print final statistics
        print("\n📊 Dataset Statistics:")
        print("=" * 60)
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Safe prompts: {stats['safe_prompts']} ({stats['safe_percentage']:.1f}%)")
        print(f"Unsafe prompts: {stats['unsafe_prompts']} ({stats['unsafe_percentage']:.1f}%)")
        print("\nSource distribution:")
        for source, count in stats['source_distribution'].items():
            print(f"  {source}: {count}")
        
        print("\n🎉 Dataset generation completed successfully!")
        return stats

def main():
    """
    Main function to run the dataset generation.
    """
    # Initialize the generator
    generator = BanglaSafetyDatasetGenerator()
    
    # Generate the dataset
    stats = generator.generate_dataset()
    
    if stats:
        print(f"\n✅ Dataset generated with {stats['total_prompts']} total prompts")
    else:
        print("❌ Dataset generation failed")

if __name__ == "__main__":
    main()
