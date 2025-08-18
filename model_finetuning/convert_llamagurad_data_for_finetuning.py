import json
import glob
from pathlib import Path
from typing import List, Dict, Any

class LlamaGuardDataConverter:
    """
    A class to convert LlamaGuard batch data into a format suitable for fine-tuning.
    """
    
    def __init__(self, batch_dir: str, output_path: str):
        """
        Initialize the converter.
        
        Args:
            batch_dir: Directory containing batch files
            output_path: Path where converted data will be saved
        """
        self.batch_dir = batch_dir
        self.output_path = output_path
        self.converted_data = []
    
    def load_batch_files(self) -> List[str]:
        """
        Get list of batch file paths.
        
        Returns:
            List of batch file paths
        """
        pattern = f"{self.batch_dir}/llama_guard_dataset_bangla_batch_*.json"
        batch_files = glob.glob(pattern)
        print(f"📁 Found {len(batch_files)} batch files")
        return batch_files
    
    def process_batch_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Process a single batch file and extract conversations.
        
        Args:
            filepath: Path to the batch file
            
        Returns:
            List of processed conversation items
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                batch = json.load(f)
            
            batch_items = []
            conversations = batch.get("conversations", [])
            
            for conv in conversations:
                prompt = conv.get("prompt_bn", "").strip()
                response = conv.get("chosen_response_bn", "").strip()
                label = conv.get("prompt_safety", "").strip()
                
                # Skip if any required field is missing
                if not prompt or not response or not label:
                    continue
                
                # Convert Bengali label to English
                label = "safe" if label == "safe" else "unsafe"
                
                item = {
                    "prompt": prompt,
                    "response": response,
                    "label": label
                }
                batch_items.append(item)
            
            print(f"✅ Processed {len(batch_items)} items from {Path(filepath).name}")
            return batch_items
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return []
    
    def convert_all_batches(self) -> List[Dict[str, Any]]:
        """
        Convert all batch files and combine the data.
        
        Returns:
            Combined list of all converted items
        """
        print("🔄 Starting batch conversion...")
        
        batch_files = self.load_batch_files()
        if not batch_files:
            print("❌ No batch files found!")
            return []
        
        all_items = []
        total_processed = 0
        
        for filepath in batch_files:
            batch_items = self.process_batch_file(filepath)
            all_items.extend(batch_items)
            total_processed += len(batch_items)
        
        print(f"📊 Total items processed: {total_processed}")
        self.converted_data = all_items
        return all_items
    
    def save_converted_data(self) -> None:
        """
        Save the converted data to the output file.
        """
        if not self.converted_data:
            print("❌ No data to save!")
            return
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.converted_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Converted data saved to: {self.output_path}")
            print(f"📊 Total items saved: {len(self.converted_data)}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            raise
    
    def save_statistics(self, stats_file: str = None) -> None:
        """
        Save dataset statistics to a JSON file.
        
        Args:
            stats_file: Path to save statistics (if None, uses default path)
        """
        try:
            # Generate statistics
            stats = self.get_statistics()
            
            # Add metadata
            stats["metadata"] = {
                "batch_directory": self.batch_dir,
                "output_file": self.output_path,
                "conversion_timestamp": str(Path().cwd()),
                "total_batch_files": len(self.load_batch_files()),
                "data_format": "prompt_response_pairs"
            }
            
            # Determine stats file path
            if stats_file is None:
                stats_file = self.output_path.replace(".json", "_statistics.json")
            
            # Create output directory if it doesn't exist
            stats_dir = Path(stats_file).parent
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            # Save statistics
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"❌ Error saving statistics: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the converted data.
        
        Returns:
            Dictionary containing statistics
        """
        if not self.converted_data:
            return {"error": "No data available"}
        
        # Count labels
        label_counts = {}
        for item in self.converted_data:
            label = item.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Calculate prompt and response lengths
        prompt_lengths = [len(item.get("prompt", "")) for item in self.converted_data]
        response_lengths = [len(item.get("response", "")) for item in self.converted_data]
        
        stats = {
            "total_items": len(self.converted_data),
            "label_distribution": label_counts,
            "prompt_length_stats": {
                "min": min(prompt_lengths) if prompt_lengths else 0,
                "max": max(prompt_lengths) if prompt_lengths else 0,
                "avg": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
            },
            "response_length_stats": {
                "min": min(response_lengths) if response_lengths else 0,
                "max": max(response_lengths) if response_lengths else 0,
                "avg": sum(response_lengths) / len(response_lengths) if response_lengths else 0
            }
        }
        
        return stats
    
    def print_statistics(self) -> None:
        """
        Print statistics about the converted data.
        """
        stats = self.get_statistics()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
        
        print("\n📊 Dataset Statistics:")
        print("=" * 40)
        print(f"Total items: {stats['total_items']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Prompt length - Min: {stats['prompt_length_stats']['min']}, Max: {stats['prompt_length_stats']['max']}, Avg: {stats['prompt_length_stats']['avg']:.1f}")
        print(f"Response length - Min: {stats['response_length_stats']['min']}, Max: {stats['response_length_stats']['max']}, Avg: {stats['response_length_stats']['avg']:.1f}")
    
    def convert_and_save(self, print_stats: bool = True, save_stats: bool = True) -> List[Dict[str, Any]]:
        """
        Complete pipeline: convert batches and save data.
        
        Args:
            print_stats: Whether to print statistics
            save_stats: Whether to save statistics to JSON file
            
        Returns:
            Converted data
        """
        print("🚀 Starting LlamaGuard data conversion...")
        print("=" * 50)
        
        # Convert all batches
        converted_data = self.convert_all_batches()
        
        if not converted_data:
            print("❌ No data to save!")
            return []
        
        # Save converted data
        self.save_converted_data()
        
        # Save statistics if requested
        if save_stats:
            self.save_statistics()
        
        # Print statistics if requested
        if print_stats:
            self.print_statistics()
        
        print("\n✅ Data conversion completed!")
        return converted_data


def main():
    """Main function to run the data conversion."""
    # Configuration
    batch_directory = "/home/malam10/projects/ai-safety-bangla/llamaguard_dataset/bangla_batches"
    output_path = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_mixed.json"
    
    # Initialize converter
    converter = LlamaGuardDataConverter(batch_directory, output_path)
    
    # Convert and save data
    converted_data = converter.convert_and_save(print_stats=True, save_stats=True)
    
    print(f"\n🎉 Data ready for fine-tuning!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Statistics file: {output_path.replace('.json', '_statistics.json')}")


if __name__ == "__main__":
    main()
