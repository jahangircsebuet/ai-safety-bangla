"""
Combine various Bangla prompt datasets into a single safety
classification dataset.

This script aggregates prompts from multiple sources (LlamaGuard
batches, MultiJail, CatQA and the Aegis dataset) and builds a
harmonised training set for a safety classifier.  Each prompt is
assigned a safety ``label`` (``"safe"`` or ``"unsafe"``) and its
originating ``source``.  The resulting collection is written to a
user‑specified location in JSON format and basic dataset statistics
are displayed.

Compared with the original script, this version adds support for the
Aegis AI Content‑Safety dataset via the ``add_aegis_data`` method.  To
use the new dataset, first generate ``converted_aegis_bangla.json``
with the ``1_4_convert_ageis_dataset.py`` script and then point this
generator to that file.
"""

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BanglaSafetyDatasetGenerator:
    """Generate a combined Bangla safety dataset from multiple sources."""

    def __init__(self, batch_dir: str = "llamaguard_dataset/bangla_batches"):
        """Create a new dataset generator.

        Args:
            batch_dir: The directory containing LlamaGuard batch files.  Each
                batch should be a JSON file with a ``conversations`` key.
        """
        self.batch_dir = batch_dir
        self.compiled_data: List[Dict[str, str]] = []

    def load_batch_files(self) -> List[str]:
        """Locate and sort all LlamaGuard batch files in ``batch_dir``.

        Returns:
            A list of file paths sorted in ascending order.
        """
        batch_pattern = f"{self.batch_dir}/llama_guard_dataset_bangla_batch_*.json"
        batch_files = sorted(glob.glob(batch_pattern))
        print(f" Found {len(batch_files)} batch files in {self.batch_dir}")
        return batch_files

    def extract_llamaguard_data(self, batch_files: List[str]) -> None:
        """Extract prompts and safety labels from LlamaGuard batch files.

        Each batch file is expected to contain a ``conversations`` list.  From
        each conversation we read ``prompt_bn`` (the Bangla prompt) and
        ``prompt_safety`` (the human/LLM‑evaluated safety label).  Only
        records where both fields are present and non‑empty are retained.

        Args:
            batch_files: A list of batch file paths.
        """
        print(" Extracting LlamaGuard data from batch files...")
        for i, file in enumerate(batch_files):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    batch = json.load(f)
            except Exception as exc:
                print(f"  ❌ Error processing {file}: {exc}")
                continue

            conversations = batch.get("conversations", [])
            extracted_count = 0
            for item in conversations:
                prompt = item.get("prompt_bn", "").strip()
                label = item.get("prompt_safety", "").strip()
                if prompt and label:
                    self.compiled_data.append(
                        {
                            "prompt": prompt,
                            "label": "safe" if label == "safe" else "unsafe",
                            "source": "llamaguard",
                        }
                    )
                    extracted_count += 1
            print(f"  ✅ Batch {i + 1}: Extracted {extracted_count} conversations")
        total_llamaguard = len([x for x in self.compiled_data if x["source"] == "llamaguard"])
        print(f" Total LlamaGuard data extracted: {total_llamaguard}")

    def add_multijail_data(self, file_path: str = "datasets/converted_multijail_bangla.json") -> None:
        """Append prompts from the MultiJail dataset to the compiled data.

        All MultiJail prompts are considered unsafe.  The JSON file is expected
        to have a top‑level key ``prompts`` containing objects with a
        ``prompt`` field.

        Args:
            file_path: Path to the converted MultiJail dataset.
        """
        print(f" Adding MultiJail data from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                multi_data = json.load(f)
        except FileNotFoundError:
            print(f"  ❌ MultiJail file not found: {file_path}")
            return
        except Exception as exc:
            print(f"  ❌ Error loading MultiJail data: {exc}")
            return

        prompts = multi_data.get("prompts", [])
        added_count = 0
        for item in prompts:
            prompt = item.get("prompt", "").strip()
            if prompt:
                self.compiled_data.append(
                    {
                        "prompt": prompt,
                        "label": "unsafe",  # all MultiJail prompts are unsafe
                        "source": "multijail",
                    }
                )
                added_count += 1
        print(f"  ✅ Added {added_count} MultiJail prompts")

    def add_catqa_data(self, file_path: str = "datasets/converted_catqa_bangla.json") -> None:
        """Append prompts from the CatQA dataset.

        CatQA prompts are also considered unsafe.  The JSON file must
        contain a top‑level ``prompts`` list of objects with a ``prompt`` field.

        Args:
            file_path: Path to the converted CatQA dataset.
        """
        print(f" Adding CatQA data from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                catqa_data = json.load(f)
        except FileNotFoundError:
            print(f"  ❌ CatQA file not found: {file_path}")
            return
        except Exception as exc:
            print(f"  ❌ Error loading CatQA data: {exc}")
            return

        prompts = catqa_data.get("prompts", [])
        added_count = 0
        for item in prompts:
            prompt = item.get("prompt", "").strip()
            if prompt:
                self.compiled_data.append(
                    {
                        "prompt": prompt,
                        "label": "unsafe",  # all CatQA prompts are unsafe
                        "source": "catqa",
                    }
                )
                added_count += 1
        print(f"  ✅ Added {added_count} CatQA prompts")

    def add_aegis_data(self, file_path: str = "datasets/converted_aegis_bangla.json") -> None:
        """Append prompts from the translated Aegis dataset.

        The converted Aegis file should have a top‑level ``prompts`` list
        where each element contains a translated ``prompt`` and a
        ``label`` field indicating whether the original prompt was safe or
        unsafe.  This method respects the provided label: prompts labelled
        as ``safe`` are added as safe, all others are considered unsafe.

        Args:
            file_path: Path to the converted Aegis dataset.
        """
        print(f" Adding Aegis data from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                aegis_data = json.load(f)
        except FileNotFoundError:
            print(f"  ❌ Aegis file not found: {file_path}")
            return
        except Exception as exc:
            print(f"  ❌ Error loading Aegis data: {exc}")
            return

        prompts = aegis_data.get("prompts", [])
        added_count = 0
        for item in prompts:
            prompt = (item.get("prompt") or "").strip()
            label = (item.get("label") or "unsafe").strip().lower()
            if prompt:
                self.compiled_data.append(
                    {
                        "prompt": prompt,
                        "label": "safe" if label == "safe" else "unsafe",
                        "source": "aegis",
                    }
                )
                added_count += 1
        print(f"  ✅ Added {added_count} Aegis prompts")

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Compute basic statistics about the compiled dataset.

        Returns:
            A dictionary containing the total number of prompts, the counts
            and percentages of safe/unsafe prompts and a breakdown by source.
        """
        total_prompts = len(self.compiled_data)
        safe_prompts = len([x for x in self.compiled_data if x["label"] == "safe"])
        unsafe_prompts = len([x for x in self.compiled_data if x["label"] == "unsafe"])
        source_counts: Dict[str, int] = {}
        for item in self.compiled_data:
            source = item.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "total_prompts": total_prompts,
            "safe_prompts": safe_prompts,
            "unsafe_prompts": unsafe_prompts,
            "safe_percentage": (safe_prompts / total_prompts * 100) if total_prompts else 0,
            "unsafe_percentage": (unsafe_prompts / total_prompts * 100) if total_prompts else 0,
            "source_distribution": source_counts,
        }

    def save_dataset(self, output_path: str = "datasets/bangla_safety_prompt_dataset.json") -> None:
        """Serialize the compiled dataset to disk.

        Args:
            output_path: The file to write the compiled dataset to.  Parent
                directories will be created if they do not exist.
        """
        print(f" Saving combined dataset to: {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(self.compiled_data, f, ensure_ascii=False, indent=2)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Dataset saved successfully! File size: {file_size_mb:.2f} MB")

    def generate_dataset(
        self,
        output_path: str = "datasets/bangla_safety_prompt_dataset.json",
        multijail_path: str = "datasets/converted_multijail_bangla.json",
        catqa_path: str = "datasets/converted_catqa_bangla.json",
        aegis_path: str = "datasets/converted_aegis_bangla.json",
    ) -> Dict[str, Any]:
        """Orchestrate the entire dataset generation process.

        This method loads LlamaGuard batches, appends MultiJail, CatQA and
        Aegis data, computes statistics and writes the final dataset to disk.

        Args:
            output_path: Where to save the combined dataset.
            multijail_path: Path to the converted MultiJail dataset.
            catqa_path: Path to the converted CatQA dataset.
            aegis_path: Path to the converted Aegis dataset.

        Returns:
            A dictionary of dataset statistics.
        """
        print(" Starting Bangla Safety Dataset Generation...")
        print("=" * 60)
        # Step 1: load LlamaGuard batch files
        batch_files = self.load_batch_files()
        if not batch_files:
            print("❌ No LlamaGuard batch files found!")
            return {}
        # Step 2: extract LlamaGuard data
        self.extract_llamaguard_data(batch_files)
        # Step 3: add MultiJail data
        self.add_multijail_data(multijail_path)
        # Step 4: add CatQA data
        self.add_catqa_data(catqa_path)
        # Step 5: add Aegis data
        self.add_aegis_data(aegis_path)
        # Step 6: compute statistics
        stats = self.get_dataset_statistics()
        # Step 7: save the compiled dataset
        self.save_dataset(output_path)
        # Print final statistics
        print("\n Dataset Statistics:")
        print("=" * 60)
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Safe prompts: {stats['safe_prompts']} ({stats['safe_percentage']:.1f}%)")
        print(f"Unsafe prompts: {stats['unsafe_prompts']} ({stats['unsafe_percentage']:.1f}%)")
        print("\nSource distribution:")
        for source, count in stats['source_distribution'].items():
            print(f"  {source}: {count}")
        print("\n Dataset generation completed successfully!")
        return stats


def main() -> None:
    """Execute the dataset generation when run as a script."""
    generator = BanglaSafetyDatasetGenerator()
    stats = generator.generate_dataset()
    if stats:
        print(f"\n✅ Dataset generated with {stats['total_prompts']} total prompts")
    else:
        print("❌ Dataset generation failed")


if __name__ == "__main__":
    main()