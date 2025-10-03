#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_safety_csv.py
Reads a CSV formatted like the dummy dataset and produces charts + a summary.

Required columns:
  prompt, response, prompt_label, response_label, prompt_category, data_source
"""

import os
import argparse
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

REQUIRED_COLS = [
    "prompt",
    "response",
    "prompt_label",
    "response_label",
    "prompt_category",
    "data_source",
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def bar_plot(series_or_df, title, xlabel, out_path, stacked_df=None, headless=False):
    plt.figure()
    if stacked_df is not None:
        stacked_df.plot(kind="bar", stacked=True)
    else:
        series_or_df.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if not headless:
        plt.show()
    plt.close()

def pie_plot(series, title, out_path, headless=False):
    plt.figure()
    series.plot(kind="pie", autopct="%1.1f%%")
    plt.title(title)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if not headless:
        plt.show()
    plt.close()

def analyze_dataset(csv_path: str, out_dir: str, top_k_categories: int = 10, headless: bool = False):
    # Set backend for headless runs (safe on servers/SSH)
    if headless:
        matplotlib.use("Agg")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate schema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Summaries
    summary = {
        "path": csv_path,
        "n_rows": len(df),
        "prompt_label_counts": df["prompt_label"].value_counts().to_dict(),
        "response_label_counts": df["response_label"].value_counts().to_dict(),
        "category_counts_topK": df["prompt_category"].value_counts().head(top_k_categories).to_dict(),
        "data_source_counts": df["data_source"].value_counts().to_dict(),
    }
    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))

    # Output dir setup
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    charts_dir = os.path.join(out_dir, base_name)
    ensure_dir(charts_dir)

    # --- BAR CHARTS ---
    bar_plot(
        df["prompt_label"].value_counts(),
        "Distribution of Prompt Labels",
        "Prompt Label",
        os.path.join(charts_dir, "bar_prompt_labels.png"),
        headless=headless,
    )

    bar_plot(
        df["response_label"].value_counts(),
        "Distribution of Response Labels",
        "Response Label",
        os.path.join(charts_dir, "bar_response_labels.png"),
        headless=headless,
    )

    bar_plot(
        df["prompt_category"].value_counts().head(top_k_categories),
        f"Top {top_k_categories} Prompt Categories",
        "Category",
        os.path.join(charts_dir, "bar_top_categories.png"),
        headless=headless,
    )

    bar_plot(
        df["data_source"].value_counts(),
        "Distribution of Data Sources",
        "Source",
        os.path.join(charts_dir, "bar_sources.png"),
        headless=headless,
    )

    # Stacked bars
    top_cats = df["prompt_category"].value_counts().head(min(top_k_categories, 8)).index
    ctab_cat = pd.crosstab(
        df.loc[df["prompt_category"].isin(top_cats), "prompt_category"],
        df.loc[df["prompt_category"].isin(top_cats), "prompt_label"],
    )
    bar_plot(
        None,
        "Prompt Label by Top Categories",
        "Category",
        os.path.join(charts_dir, "bar_stacked_prompt_by_cat.png"),
        stacked_df=ctab_cat,
        headless=headless,
    )

    ctab_src = pd.crosstab(df["data_source"], df["response_label"])
    bar_plot(
        None,
        "Response Label by Data Source",
        "Source",
        os.path.join(charts_dir, "bar_stacked_response_by_source.png"),
        stacked_df=ctab_src,
        headless=headless,
    )

    # --- PIE CHARTS ---
    pie_plot(
        df["prompt_label"].value_counts(),
        "Prompt Label Distribution (Pie)",
        os.path.join(charts_dir, "pie_prompt_labels.png"),
        headless=headless,
    )

    pie_plot(
        df["response_label"].value_counts(),
        "Response Label Distribution (Pie)",
        os.path.join(charts_dir, "pie_response_labels.png"),
        headless=headless,
    )

    pie_plot(
        df["data_source"].value_counts(),
        "Data Source Distribution (Pie)",
        os.path.join(charts_dir, "pie_sources.png"),
        headless=headless,
    )

    pie_plot(
        df["prompt_category"].value_counts().head(min(top_k_categories, 8)),
        "Top Categories (Pie)",
        os.path.join(charts_dir, "pie_top_categories.png"),
        headless=headless,
    )

    print(f"Charts saved in: {charts_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a safety CSV and produce charts.")
    parser.add_argument("--csv", required=True, help="Path to the CSV (dummy-data schema).")
    parser.add_argument("--outdir", default="charts", help="Directory where charts will be saved.")
    parser.add_argument("--topk", type=int, default=10, help="Top-K categories for charts.")
    parser.add_argument("--headless", action="store_true", help="Save-only (no GUI popups).")
    args = parser.parse_args()

    analyze_dataset(
        csv_path=args.csv,
        out_dir=args.outdir,
        top_k_categories=args.topk,
        headless=args.headless,
    )

if __name__ == "__main__":
    main()
