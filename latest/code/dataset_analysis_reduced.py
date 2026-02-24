# reduced_analysis_updated.py
# Uses df_agg produced by your flatten_agg() (already merges per dataset differences).
# Keeps ONLY:
# dataset_name, dataset_file, agg_prompt_bn, agg_response_bn,
# agg_prompt_safety, agg_harm_score_mean, selected_model

import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Save/load (JSONL recommended)
# ---------------------------

def save_df_jsonl(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

def load_df_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True)

# ---------------------------
# Robust converters
# ---------------------------

def safe_float(x):
    """Return float or np.nan for None/'None'/'null'/''/list/etc."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        try:
            return np.nan if (isinstance(x, float) and np.isnan(x)) else float(x)
        except Exception:
            return float(x)
    if isinstance(x, list):
        return safe_float(x[0]) if len(x) > 0 else np.nan
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"", "none", "null", "nan"}:
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return np.nan

def normalize_safety(x):
    """Normalize to: safe | unsafe | unknown."""
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    # aegis: prompt_label is often 'unsafe'/'safe'
    if s in {"safe", "unsafe"}:
        return s
    # allow variants
    if s in {"prompt_safe", "ok"}:
        return "safe"
    if s in {"prompt_unsafe", "harmful"}:
        return "unsafe"
    return "unknown"

# ---------------------------
# Reduced schema
# ---------------------------

REDUCED_COLS = [
    "dataset_name",
    "dataset_file",
    "agg_prompt_bn",
    "agg_response_bn",
    "agg_prompt_safety",
    "agg_harm_score_mean",
    "selected_model",
]

def ensure_reduced_df(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Takes df_agg from your flatten_agg() and returns a reduced + cleaned df.
    Works even if columns are missing in some datasets.
    """
    df = df_agg.copy()

    # Ensure columns exist
    for c in REDUCED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize types/values
    df["dataset_name"] = df["dataset_name"].fillna("UNKNOWN_DATASET").astype(str)
    df["dataset_file"] = df["dataset_file"].fillna("UNKNOWN_FILE").astype(str)
    df["selected_model"] = df["selected_model"].fillna("UNKNOWN_MODEL").astype(str)

    df["agg_prompt_safety"] = df["agg_prompt_safety"].apply(normalize_safety)

    # Numeric coercion (Aegis will become NaN -> fine)
    df["agg_harm_score_mean"] = df["agg_harm_score_mean"].apply(safe_float)

    # Clean empty strings
    df["agg_prompt_bn"] = df["agg_prompt_bn"].replace("", np.nan)
    df["agg_response_bn"] = df["agg_response_bn"].replace("", np.nan)

    return df[REDUCED_COLS].copy()

# ---------------------------
# Analyses (reduced fields only)
# ---------------------------

def analysis_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset_name"):
        for c in REDUCED_COLS:
            rows.append({
                "dataset_name": ds,
                "column": c,
                "missing_rate": float(g[c].isna().mean()),
                "n_rows": int(len(g)),
            })
    return pd.DataFrame(rows)

def analysis_counts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts_ds = df.groupby("dataset_name").size().reset_index(name="n_prompts")
    counts_model = df.groupby(["dataset_name", "selected_model"]).size().reset_index(name="n_selected")
    counts_model["pct_selected"] = counts_model.groupby("dataset_name")["n_selected"].transform(lambda x: x / x.sum())
    return counts_ds, counts_model

def analysis_safety_rates(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["dataset_name", "agg_prompt_safety"]).size().reset_index(name="count")
    t["pct"] = t.groupby("dataset_name")["count"].transform(lambda x: x / x.sum())
    return t

def analysis_harm_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only scored prompts used for mean/std/quantiles.
    Also reports coverage.
    """
    g = df.copy()
    g["has_score"] = g["agg_harm_score_mean"].notna()

    stats = g.groupby("dataset_name").apply(
        lambda x: pd.Series({
            "n_total": int(len(x)),
            "n_scored": int(x["has_score"].sum()),
            "score_coverage": float(x["has_score"].mean()),
            "mean": float(x.loc[x["has_score"], "agg_harm_score_mean"].mean()) if x["has_score"].any() else np.nan,
            "std": float(x.loc[x["has_score"], "agg_harm_score_mean"].std()) if x["has_score"].sum() > 1 else np.nan,
            "q25": float(x.loc[x["has_score"], "agg_harm_score_mean"].quantile(0.25)) if x["has_score"].any() else np.nan,
            "q50": float(x.loc[x["has_score"], "agg_harm_score_mean"].quantile(0.50)) if x["has_score"].any() else np.nan,
            "q75": float(x.loc[x["has_score"], "agg_harm_score_mean"].quantile(0.75)) if x["has_score"].any() else np.nan,
        })
    ).reset_index()

    return stats

def analysis_response_length(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["resp_len"] = g["agg_response_bn"].apply(lambda s: len(s) if isinstance(s, str) and s.strip() else np.nan)
    t = g.groupby("dataset_name")["resp_len"].agg(
        n_nonempty="count",
        mean="mean",
        std="std",
        q50=lambda x: x.quantile(0.50) if len(x) else np.nan,
        q90=lambda x: x.quantile(0.90) if len(x) else np.nan,
    ).reset_index()
    return t

# ---------------------------
# Plot saving helper
# ---------------------------

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 11})

def save_figure(name: str, out_dir: str = "figures_min", dpi: int = 300):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=dpi)
    plt.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.close()

# ---------------------------
# Plots (reduced)
# ---------------------------

def plot_selected_model_dist(sel_dist: pd.DataFrame, out_dir="figures_min"):
    plt.figure(figsize=(7,4))
    sns.barplot(data=sel_dist, x="pct_selected", y="selected_model", hue="dataset_name")
    plt.xlabel("Fraction selected (within dataset)")
    plt.ylabel("Selected model")
    plt.title("Aggregate selection distribution")
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)
    save_figure("selected_model_distribution", out_dir)

def plot_safety_rates(safety_rates: pd.DataFrame, out_dir="figures_min"):
    plt.figure(figsize=(7,4))
    sns.barplot(data=safety_rates, x="dataset_name", y="pct", hue="agg_prompt_safety")
    plt.xlabel("Dataset")
    plt.ylabel("Fraction")
    plt.title("Prompt safety label distribution")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="agg_prompt_safety", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    save_figure("prompt_safety_distribution", out_dir)

def plot_harm_distribution(df: pd.DataFrame, out_dir="figures_min"):
    g = df[df["agg_harm_score_mean"].notna()].copy()
    if g.empty:
        return
    plt.figure(figsize=(7,4))
    sns.violinplot(data=g, x="dataset_name", y="agg_harm_score_mean", inner="quartile", cut=0)
    plt.xlabel("Dataset")
    plt.ylabel("Aggregated harm score")
    plt.title("Harm score distribution (scored prompts only)")
    plt.xticks(rotation=25, ha="right")
    save_figure("harm_score_violin", out_dir)

def plot_harm_coverage(harm_stats: pd.DataFrame, out_dir="figures_min"):
    plt.figure(figsize=(7,4))
    sns.barplot(data=harm_stats, x="dataset_name", y="score_coverage")
    plt.ylim(0, 1)
    plt.xlabel("Dataset")
    plt.ylabel("Harm score coverage")
    plt.title("Fraction of prompts with a valid harm score")
    plt.xticks(rotation=25, ha="right")
    save_figure("harm_score_coverage", out_dir)

def plot_response_length(resp_stats: pd.DataFrame, out_dir="figures_min"):
    plt.figure(figsize=(7,4))
    sns.barplot(data=resp_stats, x="dataset_name", y="mean")
    plt.xlabel("Dataset")
    plt.ylabel("Mean response length (chars)")
    plt.title("Average Bangla response length")
    plt.xticks(rotation=25, ha="right")
    save_figure("response_length_mean", out_dir)

def plot_missingness_heatmap(miss: pd.DataFrame, out_dir="figures_min"):
    piv = miss.pivot(index="dataset_name", columns="column", values="missing_rate").fillna(1.0)
    plt.figure(figsize=(10,4))
    sns.heatmap(piv, vmin=0, vmax=1, cmap="Reds", annot=False)
    plt.xlabel("Column")
    plt.ylabel("Dataset")
    plt.title("Missingness heatmap (reduced fields)")
    save_figure("missingness_heatmap", out_dir)

# ---------------------------
# Runner
# ---------------------------

def run_reduced_analyses(df_agg, cache_dir="cache_min", fig_dir="figures_min") -> Dict[str, Any]:
    """
    df_agg: DataFrame from your flatten_agg(objects).
    Saves:
      - reduced df JSONL
      - analysis tables JSONL
      - plots PNG+PDF
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = ensure_reduced_df(df_agg)

    # Cache reduced df
    save_df_jsonl(df, os.path.join(cache_dir, "df_agg_reduced.jsonl"))

    out = {}
    out["A1_missingness"] = analysis_missingness(df)
    out["A2_counts_dataset"], out["A2_selected_model_dist"] = analysis_counts(df)
    out["A3_safety_rates"] = analysis_safety_rates(df)
    out["A4_harm_stats"] = analysis_harm_stats(df)
    out["A5_response_len_stats"] = analysis_response_length(df)

    # Save tables
    save_df_jsonl(out["A1_missingness"], os.path.join(cache_dir, "A1_missingness.jsonl"))
    save_df_jsonl(out["A2_counts_dataset"], os.path.join(cache_dir, "A2_counts_dataset.jsonl"))
    save_df_jsonl(out["A2_selected_model_dist"], os.path.join(cache_dir, "A2_selected_model_dist.jsonl"))
    save_df_jsonl(out["A3_safety_rates"], os.path.join(cache_dir, "A3_safety_rates.jsonl"))
    save_df_jsonl(out["A4_harm_stats"], os.path.join(cache_dir, "A4_harm_stats.jsonl"))
    save_df_jsonl(out["A5_response_len_stats"], os.path.join(cache_dir, "A5_response_len_stats.jsonl"))

    # Plots
    plot_selected_model_dist(out["A2_selected_model_dist"], out_dir=fig_dir)
    plot_safety_rates(out["A3_safety_rates"], out_dir=fig_dir)
    plot_harm_distribution(df, out_dir=fig_dir)
    plot_harm_coverage(out["A4_harm_stats"], out_dir=fig_dir)
    plot_response_length(out["A5_response_len_stats"], out_dir=fig_dir)
    plot_missingness_heatmap(out["A1_missingness"], out_dir=fig_dir)

    out["df_agg_reduced"] = df
    return out
