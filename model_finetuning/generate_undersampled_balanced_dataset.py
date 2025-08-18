import json
import pandas as pd
from cluster_based_under_sampler import ClusterCentroidUnderSampler
from random_under_sampler import RandomUnderSamplerBalancer

# Load imbalanced data file
with open("/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_mixed.json", "r", encoding="utf-8") as f:
    imbalanced = json.load(f)

# 1) Random undersampling
rus = RandomUnderSamplerBalancer(seed=2025)
balanced_random = rus.balance(imbalanced)

# Save to JSON
with open("/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_balanced_random.json", "w", encoding="utf-8") as f:
    json.dump(balanced_random, f, ensure_ascii=False, indent=2)

# 2) Cluster-based undersampling
ccus = ClusterCentroidUnderSampler(seed=2025)
balanced_cluster = ccus.balance(imbalanced)

with open("/home/malam10/projects/ai-safety-bangla/datasets/bangla_prompt_response_balanced_cluster.json", "w", encoding="utf-8") as f:
    json.dump(balanced_cluster, f, ensure_ascii=False, indent=2)

print("Random balance count:", len(balanced_random))
print("Cluster balance count:", len(balanced_cluster))
print("Labels:", pd.Series([r['label'] for r in balanced_cluster]).value_counts())