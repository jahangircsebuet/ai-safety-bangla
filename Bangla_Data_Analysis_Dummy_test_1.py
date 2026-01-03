#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Generate dummy safety data and/or analyze an existing CSV.")
parser.add_argument("--csv", type=str, default="", help="Path to a CSV with columns: prompt,response,prompt_label,response_label,prompt_category,data_source")
parser.add_argument("--rows", type=int, default=315, help="If --csv not given, generate this many dummy rows.")
parser.add_argument("--headless", action="store_true", help="Do not call plt.show(); save figures only.")
parser.add_argument("--topk", type=int, default=10, help="Top-K categories in charts.")
args = parser.parse_args()

# If headless requested, force non-GUI backend
if args.headless:
    matplotlib.use("Agg")  # safe for servers/SSH

# ---------- Helpers ----------
REQUIRED_COLS = ["prompt","response","prompt_label","response_label","prompt_category","data_source"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def bar_plot(series, title, xlabel, out_path, stacked_df=None):
    plt.figure()
    if stacked_df is not None:
        stacked_df.plot(kind="bar", stacked=True)
    else:
        series.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if not args.headless:
        plt.show()
    plt.close()

def pie_plot(series, title, out_path):
    plt.figure()
    series.plot(kind="pie", autopct="%1.1f%%")
    plt.title(title)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if not args.headless:
        plt.show()
    plt.close()

# ---------- Dummy generation ----------
categories = [
    "violence","sexual","malware","hate_speech","self_harm",
    "misinformation","harassment","fraud","child_safety",
    "weapons","financial_crime","privacy","medical_misinformation",
    "career","extremism","education","politics","drugs","terrorism","other"
]
sources = ["multijail","catqa","anthropic","aegis","hatespeech"]

prompt_pool = [
    "ডামি প্রম্পট নম্বর {i} — এখানে একটি প্রশ্ন থাকবে।",
    "উদাহরণ প্রম্পট {i}: পরীক্ষামূলক ডেটার জন্য একটি বাক্য।",
    "সিমুলেটেড ইনপুট {i} — বিশ্লেষণের সুবিধার্থে।",
    "নমুনা প্রশ্ন {i} — কেবলমাত্র ডামি ডেটা।",
    "প্রসঙ্গ {i} — গবেষণার উদ্দেশ্যে কাল্পনিক ইনপুট।"
]
refusal_pool = [
    "দুঃখিত, আমি এই অনুরোধে সহায়তা করতে পারি না।",
    "ক্ষমাপ্রার্থী, এই বিষয়ে নির্দেশনা দেওয়া সম্ভব নয়।",
    "আমি অনৈতিক/ক্ষতিকর কার্যকলাপে সহায়তা করি না।",
    "অনুগ্রহ করে নিরাপদ ও আইনসম্মত পন্থা অনুসরণ করুন।"
]
benign_pool = [
    "এটি একটি ডামি উত্তর, যেখানে নিরাপদ প্রতিক্রিয়া প্রদান করা হয়েছে।",
    "ডেমো উদ্দেশ্যে তৈরি একটি সুসংগত, নিরাপদ উত্তর।",
    "উত্তরটি কেবল পরীক্ষামূলক এবং নিরাপদ।",
    "এখানে একটি সাধারণ, নিরপেক্ষ প্রতিক্রিয়া দেওয়া হলো।"
]

def generate_dummy(n_rows=315, seed=42):
    random.seed(seed); np.random.seed(seed)
    rows = []
    for i in range(1, n_rows+1):
        prompt_label = random.choices(["unsafe","safe"], weights=[0.52,0.48])[0]
        response_label = "safe" if random.random() < 0.75 else "unsafe"
        rows.append({
            "prompt": random.choice(prompt_pool).format(i=i),
            "response": random.choice(refusal_pool) if prompt_label=="unsafe" else random.choice(benign_pool),
            "prompt_label": prompt_label,
            "response_label": response_label,
            "prompt_category": random.choice(categories),
            "data_source": random.choice(sources),
        })
    return pd.DataFrame(rows)

# ---------- Load or create data ----------
if args.csv:
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    base_name = os.path.splitext(os.path.basename(args.csv))[0]
else:
    df = generate_dummy(n_rows=args.rows)
    base_name = f"dummy_multijail_like_{args.rows}"
    df.to_csv(f"{base_name}.csv", index=False)
    with open(f"{base_name}.json","w",encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"Wrote: {base_name}.csv and {base_name}.json")

print(f"Rows: {len(df)}")

# ---------- Charts ----------
charts_dir = os.path.join("charts", base_name)
ensure_dir(charts_dir)

# Summaries
print("Prompt labels:\n", df["prompt_label"].value_counts())
print("Response labels:\n", df["response_label"].value_counts())
print("Top categories:\n", df["prompt_category"].value_counts().head(args.topk))
print("Data sources:\n", df["data_source"].value_counts())

# Bar charts
bar_plot(df["prompt_label"].value_counts(), "Distribution of Prompt Labels", "Prompt Label",
         os.path.join(charts_dir,"bar_prompt_labels.png"))

bar_plot(df["response_label"].value_counts(), "Distribution of Response Labels", "Response Label",
         os.path.join(charts_dir,"bar_response_labels.png"))

bar_plot(df["prompt_category"].value_counts().head(args.topk), f"Top {args.topk} Prompt Categories", "Category",
         os.path.join(charts_dir,"bar_top_categories.png"))

bar_plot(df["data_source"].value_counts(), "Distribution of Data Sources", "Source",
         os.path.join(charts_dir,"bar_sources.png"))

# Stacked bars
top_cats = df["prompt_category"].value_counts().head(min(args.topk,8)).index
ctab = pd.crosstab(df.loc[df["prompt_category"].isin(top_cats),"prompt_category"],
                   df.loc[df["prompt_category"].isin(top_cats),"prompt_label"])
bar_plot(None, "Prompt Label by Top Categories", "Category",
         os.path.join(charts_dir,"bar_stacked_prompt_by_cat.png"), stacked_df=ctab)

ctab2 = pd.crosstab(df["data_source"], df["response_label"])
bar_plot(None, "Response Label by Data Source", "Source",
         os.path.join(charts_dir,"bar_stacked_response_by_source.png"), stacked_df=ctab2)

# Pie charts
pie_plot(df["prompt_label"].value_counts(), "Prompt Label Distribution (Pie)",
         os.path.join(charts_dir,"pie_prompt_labels.png"))

pie_plot(df["response_label"].value_counts(), "Response Label Distribution (Pie)",
         os.path.join(charts_dir,"pie_response_labels.png"))

pie_plot(df["data_source"].value_counts(), "Data Source Distribution (Pie)",
         os.path.join(charts_dir,"pie_sources.png"))

pie_plot(df["prompt_category"].value_counts().head(min(args.topk,8)), "Top Categories (Pie)",
         os.path.join(charts_dir,"pie_top_categories.png"))

print(f"Charts saved in: {charts_dir}")
print("Tip: If you're on a server/SSH, use --headless and open the PNGs.")
