"""
LCDM: Lexical Cultural Density Metric (prompt-only) for Bangla cultural lexical analysis
-------------------------------------------------------------------------------------

What this script does (end-to-end):
1) Loads one or more datasets (MultiJail, CatQA, Aegis, Hatespeech, Toxic) from JSON/JSONL/CSV.
2) Extracts Bangla prompts from configurable field names (e.g., prompt_bn, prompt, text).
3) Performs prompt-only lexicon matching (normalized string match).
4) Computes per-prompt:
   - cultural_lexicon_found (matched terms)
   - matched_groups (lexicon groups)
   - dominant category (single label)
   - lcdm_score in [0,1]
5) Produces dataset-level tables:
   - Avg LCDM, Std, % culturally grounded
6) Produces plots (matplotlib only; no seaborn):
   - Figure 1: Avg LCDM bar plot with error bars
   - Figure 2: Stacked bar chart of category distribution per dataset
   - Figure 3: Box plot of LCDM distributions per dataset
7) Cultural drift across datasets (pairwise):
   - Δ mean LCDM
   - Jensen–Shannon divergence between LCDM distributions

Dependencies:
  pip install pandas numpy matplotlib

Usage examples:
  python lcdm_analysis.py \
    --inputs MultiJail=multijail.jsonl CatQA=catqa.json Aegis=aegis.json Hatespeech=hs.csv Toxic=toxic.jsonl \
    --prompt_fields prompt_bn,prompt,text \
    --out_dir outputs

Notes:
- This is purely lexical: no model calls, no response analysis.
- Lexicon below is a starter; expand it with your curated terms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) Lexicon (starter)
# -----------------------------
LEXICON: Dict[str, List[str]] = {
    "Religious": ["আল্লাহ", "ঈদ", "নামাজ", "রোজা", "কোরআন", "মসজিদ", "ইসলাম", "হিন্দু", "মুসলিম", "খ্রিস্টান"],
    "Honor_Shame": ["মান-সম্মান", "ইজ্জত", "সম্মান", "অপমান", "লজ্জা", "সম্মানহানি", "মানহানি", "বেইজ্জতি"],
    "Kinship": ["মা", "বাবা", "ভাই", "বোন", "চাচা", "খালা", "দাদা", "দাদি", "নানা", "নানি", "শ্বশুর", "শাশুড়ি"],
    "Family_Gender": ["বউ", "স্বামী", "স্ত্রী", "তালাক", "বিয়ে", "বিবাহ", "যৌতুক", "পরকীয়া", "পর্দা"],
    "Political": ["আওয়ামী লীগ", "বিএনপি", "ভোট", "নির্বাচন", "সরকার", "বিরোধী দল"],
    "Community_Caste": ["জাত", "বংশ", "গোত্র", "সম্প্রদায়", "বর্ণ"],
    "Bangla_Slang_Abuse": ["হারামি", "বেজন্মা", "শালা", "কুত্তা", "খানকি", "মাগী", "চোদা", "ফকিন্নি"],
}

# Map groups -> paper taxonomy categories (single-label outputs)
GROUP_TO_CATEGORY: Dict[str, str] = {
    "Religious": "Religious / Caste / Ethnic Harm",
    "Community_Caste": "Religious / Caste / Ethnic Harm",
    "Honor_Shame": "Honor / Shame--Based Harm",
    "Kinship": "Sensitive Social-Context Harm (family, marriage, gender norms)",
    "Family_Gender": "Sensitive Social-Context Harm (family, marriage, gender norms)",
    "Political": "Political / Cultural Conflict Harm",
    "Bangla_Slang_Abuse": "Bangla-Specific Abuse Patterns (slang, idioms, Banglish)",
}

# Priority order for dominant category tie-breaks (high -> low)
GROUP_PRIORITY: List[str] = [
    "Honor_Shame",
    "Religious",
    "Community_Caste",
    "Family_Gender",
    "Kinship",
    "Political",
    "Bangla_Slang_Abuse",
]


# -----------------------------
# 2) Normalization utilities
# -----------------------------
_ZW_CHARS = "\u200b\u200c\u200d\ufeff"  # ZWSP, ZWNJ, ZWJ, BOM

def normalize_bn(text: str) -> str:
    """Lightweight normalization for Bangla lexical matching."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\r", " ").replace("\n", " ")
    # remove zero-width chars
    s = re.sub(f"[{_ZW_CHARS}]", "", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # keep punctuation; but normalize hyphen variants to "-"
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return s

def compile_lexicon(lex: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """
    Compile per-group regex patterns with conservative word-boundary-like behavior.
    Bangla doesn't have perfect word boundaries for all cases, so we:
      - match term as substring, but try to avoid matching inside longer Latin strings.
    """
    compiled: Dict[str, List[re.Pattern]] = {}
    for group, terms in lex.items():
        pats = []
        for t in terms:
            t_norm = normalize_bn(t)
            # Escape regex, then allow optional surrounding whitespace/punctuation.
            # We'll just use substring match via regex search.
            pat = re.compile(re.escape(t_norm))
            pats.append(pat)
        compiled[group] = pats
    return compiled


# -----------------------------
# 3) LCDM scoring
# -----------------------------
@dataclass
class LCDMConfig:
    group_weight: float = 0.25
    repetition_bonus: float = 0.10
    score_cap: float = 1.0
    min_grounded_score: float = 0.0  # for "grounded" boolean threshold you can set > 0
    # If a prompt matches lexicon terms but only 1 group, you may want a minimum base score:
    base_if_any_match: float = 0.25

def lcdm_for_prompt(prompt: str,
                    compiled_lex: Dict[str, List[re.Pattern]],
                    cfg: LCDMConfig) -> Dict:
    """
    Compute prompt-only lexical cultural density metrics for a single prompt.
    Returns a dict containing:
      - lcdm_score
      - cultural_lexicon_found
      - matched_groups
      - dominant_category
    """
    p = normalize_bn(prompt)
    if not p:
        return {
            "lcdm_score": 0.0,
            "cultural_lexicon_found": [],
            "matched_groups": [],
            "dominant_category": "None / No Cultural Harm",
        }

    found_terms: List[str] = []
    group_hits: Dict[str, int] = {g: 0 for g in compiled_lex.keys()}
    term_hits: Dict[str, int] = {}

    for group, patterns in compiled_lex.items():
        for i, pat in enumerate(patterns):
            # find all non-overlapping matches
            matches = list(pat.finditer(p))
            if matches:
                # retrieve original term string via pattern (we only store the escaped version)
                # We'll approximate by using pat.pattern (escaped); keep readable with unescape-like:
                term = re.sub(r"\\", "", pat.pattern)  # fine for our simple escaping
                count = len(matches)
                group_hits[group] += count
                term_hits[term] = term_hits.get(term, 0) + count

    matched_groups = [g for g, c in group_hits.items() if c > 0]
    # list of unique terms found (keep stable order by decreasing count then name)
    found_terms = [t for t, _ in sorted(term_hits.items(), key=lambda x: (-x[1], x[0]))]

    # dominant group selection:
    dominant_group = None
    if matched_groups:
        # highest count, tie-break by priority list then first occurrence
        # build candidates
        max_count = max(group_hits[g] for g in matched_groups)
        candidates = [g for g in matched_groups if group_hits[g] == max_count]
        if len(candidates) == 1:
            dominant_group = candidates[0]
        else:
            # priority tie-break
            for g in GROUP_PRIORITY:
                if g in candidates:
                    dominant_group = g
                    break
            if dominant_group is None:
                dominant_group = sorted(candidates)[0]

    dominant_category = "None / No Cultural Harm"
    if dominant_group:
        dominant_category = GROUP_TO_CATEGORY.get(dominant_group, dominant_group)

    # score: breadth + repetition depth
    if not matched_groups:
        score = 0.0
    else:
        breadth = len(matched_groups) * cfg.group_weight
        # repetition: total matches beyond 1 per term
        total_repeat = 0
        for term, c in term_hits.items():
            if c > 1:
                total_repeat += (c - 1)
        depth = total_repeat * cfg.repetition_bonus
        score = max(cfg.base_if_any_match, breadth + depth)
        score = min(cfg.score_cap, score)

    return {
        "lcdm_score": float(score),
        "cultural_lexicon_found": found_terms,
        "matched_groups": matched_groups,
        "dominant_category": dominant_category,
    }


# -----------------------------
# 4) Data loading
# -----------------------------
def load_any(path: str) -> pd.DataFrame:
    """Load JSON, JSONL, or CSV into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in [".jsonl", ".jsonl.gz"]:
        rows = []
        opener = open
        if ext.endswith(".gz"):
            import gzip
            opener = gzip.open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # accept list or dict with a top-level key
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            # try common keys
            for k in ["data", "items", "records"]:
                if k in obj and isinstance(obj[k], list):
                    return pd.DataFrame(obj[k])
            # otherwise flatten dict
            return pd.json_normalize(obj)
        raise ValueError("Unsupported JSON structure")

    if ext == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported extension: {ext}")


def pick_prompt_column(df: pd.DataFrame, prompt_fields: List[str]) -> str:
    """Choose first available prompt field from candidate list."""
    for f in prompt_fields:
        if f in df.columns:
            return f
    raise ValueError(f"None of the prompt fields found: {prompt_fields}. Columns: {list(df.columns)[:30]}...")


# -----------------------------
# 5) Dataset analysis
# -----------------------------
def analyze_dataset(name: str,
                    df: pd.DataFrame,
                    prompt_col: str,
                    compiled_lex: Dict[str, List[re.Pattern]],
                    cfg: LCDMConfig) -> pd.DataFrame:
    """Return per-prompt LCDM results as a DataFrame."""
    out = df.copy()
    prompts = out[prompt_col].astype(str).fillna("")
    results = [lcdm_for_prompt(p, compiled_lex, cfg) for p in prompts]

    out["dataset"] = name
    out["prompt_col_used"] = prompt_col
    out["lcdm_score"] = [r["lcdm_score"] for r in results]
    out["dominant_category"] = [r["dominant_category"] for r in results]
    out["matched_groups"] = [r["matched_groups"] for r in results]
    out["cultural_lexicon_found"] = [r["cultural_lexicon_found"] for r in results]

    # culturally grounded flag (prompt has any matched group)
    out["is_culturally_grounded"] = out["lcdm_score"] > 0.0
    return out


def dataset_summary_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Avg LCDM, Std, % grounded."""
    g = all_rows.groupby("dataset")["lcdm_score"]
    grounded = all_rows.groupby("dataset")["is_culturally_grounded"].mean() * 100.0

    summary = pd.DataFrame({
        "avg_lcdm": g.mean(),
        "std_lcdm": g.std(ddof=0),
        "pct_culturally_grounded": grounded,
        "n": g.size(),
    }).reset_index().sort_values("avg_lcdm", ascending=True)
    return summary


def category_distribution(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Figure 2 data: proportion of dominant categories per dataset."""
    ct = pd.crosstab(all_rows["dataset"], all_rows["dominant_category"], normalize="index")
    ct = ct.reset_index()
    return ct


# -----------------------------
# 6) Drift metrics
# -----------------------------
def _hist_prob(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, density=False)
    p = h.astype(float)
    if p.sum() == 0:
        return p
    return p / p.sum()

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JSD(p||q) with log2; result in [0,1] approximately."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def cultural_drift_table(all_rows: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """
    Pairwise drift across datasets:
      - delta_mean: mean LCDM difference
      - jsd: Jensen–Shannon divergence between LCDM distributions
    """
    datasets = sorted(all_rows["dataset"].unique().tolist())
    # fixed bins for comparability
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    rows = []
    for i, a in enumerate(datasets):
        xa = all_rows.loc[all_rows["dataset"] == a, "lcdm_score"].to_numpy()
        pa = _hist_prob(xa, bin_edges)
        mean_a = float(np.mean(xa)) if len(xa) else 0.0

        for b in datasets[i + 1:]:
            xb = all_rows.loc[all_rows["dataset"] == b, "lcdm_score"].to_numpy()
            pb = _hist_prob(xb, bin_edges)
            mean_b = float(np.mean(xb)) if len(xb) else 0.0

            rows.append({
                "dataset_a": a,
                "dataset_b": b,
                "delta_mean_lcdm_(b-a)": mean_b - mean_a,
                "jsd_lcdm": jensen_shannon_divergence(pa, pb),
            })

    return pd.DataFrame(rows).sort_values(["jsd_lcdm", "delta_mean_lcdm_(b-a)"], ascending=False)


# -----------------------------
# 7) Plotting (matplotlib only)
# -----------------------------
def plot_avg_lcdm(summary: pd.DataFrame, out_path: str):
    """Figure 1: bar plot of avg LCDM with std error bars."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["avg_lcdm"].values, yerr=summary["std_lcdm"].values, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"].values, rotation=20, ha="right")
    ax.set_ylabel("Average LCDM")
    ax.set_title("Average Lexical Cultural Density (Prompt-Only)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_category_stacked(dist: pd.DataFrame, out_path: str):
    """Figure 2: stacked bar chart of category proportions per dataset."""
    # dist: crosstab normalized with dataset column
    df = dist.set_index("dataset")
    categories = [c for c in df.columns if c != "dataset"]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    bottom = np.zeros(len(df))
    x = np.arange(len(df.index))

    for cat in categories:
        vals = df[cat].values
        ax.bar(x, vals, bottom=bottom, label=cat)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df.index.values, rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Dominant Cultural Category Distribution (Prompt-Only)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_lcdm_box(all_rows: pd.DataFrame, out_path: str):
    """Figure 3: boxplot distribution of LCDM scores across datasets."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    datasets = sorted(all_rows["dataset"].unique().tolist())
    data = [all_rows.loc[all_rows["dataset"] == d, "lcdm_score"].values for d in datasets]
    ax.boxplot(data, labels=datasets, showfliers=False)
    ax.set_ylabel("LCDM score")
    ax.set_title("LCDM Distribution Across Datasets (Prompt-Only)")
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_drift_heatmap(drift_df: pd.DataFrame, out_path: str):
    """
    Optional: heatmap-style plot for JSD drift.
    Uses imshow without seaborn.
    """
    datasets = sorted(set(drift_df["dataset_a"]).union(set(drift_df["dataset_b"])))
    idx = {d: i for i, d in enumerate(datasets)}
    n = len(datasets)
    mat = np.zeros((n, n), dtype=float)

    for _, r in drift_df.iterrows():
        i = idx[r["dataset_a"]]
        j = idx[r["dataset_b"]]
        v = float(r["jsd_lcdm"])
        mat[i, j] = v
        mat[j, i] = v

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticklabels(datasets)
    ax.set_title("Cultural Drift Heatmap (JSD over LCDM distributions)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# 8) Main CLI
# -----------------------------
def parse_inputs(items: List[str]) -> Dict[str, str]:
    """
    Parse inputs like:
      ["MultiJail=path.jsonl", "CatQA=path.json", ...]
    """
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad --inputs item: {it}. Use Name=path")
        name, path = it.split("=", 1)
        out[name.strip()] = path.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="List of datasetName=path items.")
    ap.add_argument("--prompt_fields", default="prompt_bn,prompt,text",
                    help="Comma-separated candidate columns for Bangla prompt.")
    ap.add_argument("--out_dir", default="outputs", help="Output directory.")
    ap.add_argument("--bins", type=int, default=20, help="Bins for drift JSD.")
    ap.add_argument("--save_per_prompt_csv", action="store_true",
                    help="If set, saves per-prompt LCDM results for all datasets.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = parse_inputs(args.inputs)
    prompt_fields = [x.strip() for x in args.prompt_fields.split(",") if x.strip()]

    compiled_lex = compile_lexicon(LEXICON)
    cfg = LCDMConfig()

    all_results = []
    for ds_name, path in inputs.items():
        df = load_any(path)
        prompt_col = pick_prompt_column(df, prompt_fields)
        res = analyze_dataset(ds_name, df, prompt_col, compiled_lex, cfg)
        all_results.append(res)

    all_rows = pd.concat(all_results, ignore_index=True)

    # --- Tables
    summary = dataset_summary_table(all_rows)
    summary_path = os.path.join(args.out_dir, "table_dataset_lcdm_summary.csv")
    summary.to_csv(summary_path, index=False)

    dist = category_distribution(all_rows)
    dist_path = os.path.join(args.out_dir, "table_category_distribution.csv")
    dist.to_csv(dist_path, index=False)

    drift = cultural_drift_table(all_rows, bins=args.bins)
    drift_path = os.path.join(args.out_dir, "table_cultural_drift_pairwise.csv")
    drift.to_csv(drift_path, index=False)

    if args.save_per_prompt_csv:
        per_prompt_path = os.path.join(args.out_dir, "per_prompt_lcdm_results.csv")
        # keep a compact subset (you can add id columns as needed)
        keep_cols = ["dataset", "prompt_col_used", "lcdm_score", "dominant_category",
                     "matched_groups", "cultural_lexicon_found"]
        # also keep original prompt if available:
        # (we don't know the original col per row after concat, so store a unified prompt column)
        # We'll rebuild it: the prompt is stored in the original df, but easier:
        # If your datasets have standardized prompt column, add it here.
        # Otherwise, uncomment next line and ensure your datasets share a column name.
        # keep_cols.append("prompt_bn")
        all_rows[keep_cols].to_csv(per_prompt_path, index=False)

    # --- Plots
    plot_avg_lcdm(summary, os.path.join(args.out_dir, "fig1_avg_lcdm.png"))
    plot_category_stacked(dist, os.path.join(args.out_dir, "fig2_category_stacked.png"))
    plot_lcdm_box(all_rows, os.path.join(args.out_dir, "fig3_lcdm_box.png"))
    plot_drift_heatmap(drift, os.path.join(args.out_dir, "fig4_drift_jsd_heatmap.png"))

    # Print quick paths
    print("Saved:")
    print(" -", summary_path)
    print(" -", dist_path)
    print(" -", drift_path)
    print(" -", os.path.join(args.out_dir, "fig1_avg_lcdm.png"))
    print(" -", os.path.join(args.out_dir, "fig2_category_stacked.png"))
    print(" -", os.path.join(args.out_dir, "fig3_lcdm_box.png"))
    print(" -", os.path.join(args.out_dir, "fig4_drift_jsd_heatmap.png"))


if __name__ == "__main__":
    main()
