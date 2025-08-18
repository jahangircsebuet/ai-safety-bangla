import os
import json
from generate_balanced_dataset_stats import (
    plot_graphable_stats,
    plot_pca_tsne_embeddings
)

def load_json(path):
    """Loads JSON file containing list of dicts."""
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)

def main():
    # --- Paths ---
    original_fp = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_mixed.json"
    random_fp   = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_balanced_random.json"
    cluster_fp  = "/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_balanced_cluster.json"
    plot_base   = "/home/malam10/projects/ai-safety-bangla/3/plots"

    # --- Load data ---
    print("📥 Loading datasets...")
    data_original = load_json(original_fp)
    data_random   = load_json(random_fp)
    data_cluster  = load_json(cluster_fp)
    print(f"✅ Original: {len(data_original)} items, "
          f"Random-U: {len(data_random)}, Cluster-U: {len(data_cluster)}")

    # --- Comparison 1: Original vs Random Under-Sampled ---
    comp1_dir = os.path.join(plot_base, "orig_vs_random")
    print(f"\n📊 Generating plots: Original vs Random sampling → {comp1_dir}")
    os.makedirs(comp1_dir, exist_ok=True)
    plot_graphable_stats(
        data=data_original,
        output_dir=os.path.join(comp1_dir, "graphable_stats")
    )
    plot_pca_tsne_embeddings(
        data_before=data_original,
        data_after=data_random,
        output_dir=os.path.join(comp1_dir, "embeddings")
    )

    # --- Comparison 2: Original vs Cluster Centroid Under-Sampled ---
    comp2_dir = os.path.join(plot_base, "orig_vs_cluster")
    print(f"\n📊 Generating plots: Original vs Cluster sampling → {comp2_dir}")
    os.makedirs(comp2_dir, exist_ok=True)
    plot_graphable_stats(
        data=data_original,
        output_dir=os.path.join(comp2_dir, "graphable_stats")
    )
    plot_pca_tsne_embeddings(
        data_before=data_original,
        data_after=data_cluster,
        output_dir=os.path.join(comp2_dir, "embeddings")
    )

    print("\n🎉 All comparisons completed. Plots are generated inside:")
    print(f"  • {comp1_dir}")
    print(f"  • {comp2_dir}")

if __name__ == "__main__":
    main()
