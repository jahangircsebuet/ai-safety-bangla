import json
import argparse
from collections import Counter, defaultdict

# Dataset Preparation Pipeline
# toxic_prompt_dataset_preparation.py
# │
# ├── Step 1: Deduplicate prompts
# ├── Step 2: Add AEGIS category name
# ├── Step 3: Distribution analysis
# ├── Step 4: Filter by harm_score ≥ threshold
# └── Step 5: Save stats


# 🔹 Step 1: Deduplicate by text
def deduplicate_by_text(data):
    seen = set()
    unique = []
    duplicates = []

    for item in data:
        text = item.get("text", "").strip()
        if text in seen:
            duplicates.append(item)
        else:
            seen.add(text)
            unique.append(item)

    return unique, duplicates


# 🔹 Step 2: Add AEGIS category name
AEGIS_CATEGORY_MAP = {
    "S1": "Violence",
    "S2": "Sexual",
    "S3": "Criminal Planning/Confessions",
    "S4": "Guns and Illegal Weapons",
    "S5": "Controlled/Regulated Substances",
    "S6": "Suicide and Self Harm",
    "S7": "Sexual (minor)",
    "S8": "Hate/Identity Hate",
    "S9": "PII/Privacy",
    "S10": "Harassment",
    "S11": "Threat",
    "S12": "Profanity",
    "S13": "Needs Caution",
    "S14": "Other",
    "S15": "Manipulation",
    "S16": "Fraud / Deception",
    "S17": "Malware",
    "S18": "High-Risk Government Decision Making",
    "S19": "Political / Misinformation / Conspiracy",
    "S20": "Copyright / Trademark / Plagiarism",
    "S21": "Unauthorized Advice",
    "S22": "Illegal Activity",
    "S23": "Immoral / Unethical",
}

def add_aegis_category_name(data):
    for item in data:
        code = item.get("aegis_category")
        item["aegis_category_name"] = AEGIS_CATEGORY_MAP.get(code, "UNKNOWN")
    return data


# 🔹 Step 3: Data Distribution
def compute_distributions(data):
    harm_score_dist = Counter()
    aegis_code_dist = Counter()
    aegis_name_dist = Counter()

    for item in data:
        harm_score_dist[str(item.get("harm_score", "0.0"))] += 1
        aegis_code_dist[item.get("aegis_category", "UNKNOWN")] += 1
        aegis_name_dist[item.get("aegis_category_name", "UNKNOWN")] += 1

    total = len(data)
    aegis_name_percent = {
        k: round((v / total) * 100, 2)
        for k, v in aegis_name_dist.items()
    }

    return {
        "total_items": total,
        "harm_score_distribution": dict(harm_score_dist),
        "aegis_category_distribution": dict(aegis_code_dist),
        "aegis_category_name_distribution": dict(aegis_name_dist),
        "aegis_category_name_percentage_distribution": aegis_name_percent,
    }



# 🔹 Step 4: Filter by harm_score ≥ threshold
def filter_by_harm_score(data, threshold=0.7):
    return [
        item for item in data
        if float(item.get("harm_score", 0.0)) >= threshold
    ]


def main(args):
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.dedup:
        data, duplicates = deduplicate_by_text(data)
        with open("duplicates.json", "w", encoding="utf-8") as f:
            json.dump(duplicates, f, ensure_ascii=False, indent=2)

    if args.add_aegis_name:
        data = add_aegis_category_name(data)

    if args.filter_harm:
        data = filter_by_harm_score(data, args.harm_threshold)

    if args.distribution:
        stats = compute_distributions(data)
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Toxic Prompt Dataset Pipeline")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-out", default="distribution_stats.json")

    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--add-aegis-name", action="store_true")
    parser.add_argument("--distribution", action="store_true")

    parser.add_argument("--filter-harm", action="store_true")
    parser.add_argument("--harm-threshold", type=float, default=0.7)

    args = parser.parse_args()
    main(args)