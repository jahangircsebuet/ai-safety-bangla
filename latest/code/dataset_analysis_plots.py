import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11
})


# ================================
# A1. Dataset coverage & completeness
# ================================
# Plot: bar plot of per-source coverage (per dataset)

def plot_dataset_coverage(coverage_table, out_dir):
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=coverage_table,
        x="dataset_file",
        y="per_source_coverage",
        color="steelblue"
    )
    plt.ylim(0,1)
    plt.ylabel("Per-source coverage")
    plt.xlabel("Dataset")
    plt.xticks(rotation=30, ha="right")
    plt.title("Per-dataset annotation coverage")
    save_figure("selected_model_distribution", out_dir)

# 📌 Main paper (small) or appendix

# ================================
# A2. Selected model distribution
# ================================
# Plot: horizontal bar (selection bias)

def plot_selected_model_distribution(selected_dist, out_dir):
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=selected_dist,
        y="selected_model",
        x="pct",
        color="darkorange"
    )
    plt.xlabel("Fraction selected as aggregate")
    plt.ylabel("Model")
    plt.title("Aggregate label selection by model")
    save_figure("plot_selected_model_distribution", out_dir)
# 📌 Main paper (very reviewer-relevant)

# ================================
# A3. Harm score distributions (agg)
# ================================
# Plot: violin plot (severity profile)

def plot_agg_harm_distribution(df_agg, out_dir):
    plt.figure(figsize=(6,4))
    sns.violinplot(
        data=df_agg,
        x="dataset_name",
        y="agg_harm_score_mean",
        inner="quartile",
        cut=0
    )
    plt.ylabel("Aggregated harm score")
    plt.xlabel("Dataset")
    plt.title("Harm severity distribution (aggregated)")
    save_figure("plot_agg_harm_distribution", out_dir)
# 📌 Main paper


# ================================
# A4. AEGIS category distribution
# ================================
# Plot: stacked bar (top categories)

def plot_agg_category_distribution(agg_cat_top, out_dir):
    pivot = agg_cat_top.pivot(
        index="dataset_name",
        columns="agg_aegis_category",
        values="pct"
    ).fillna(0)

    pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(7,4),
        colormap="tab20"
    )
    plt.ylabel("Fraction of prompts")
    plt.xlabel("Dataset")
    plt.title("AEGIS category distribution (aggregated)")
    plt.legend(bbox_to_anchor=(1.05,1), title="AEGIS")
    save_figure("plot_agg_category_distribution", out_dir)
# 📌 Main paper

# ================================
# A5. Prompt safety disagreement
# ================================
# Plot: bar of disagreement rate

def plot_safety_disagreement(safety_disagree_table, out_dir):
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=safety_disagree_table,
        x="dataset_file",
        y="disagreement_rate",
        color="crimson"
    )
    plt.ylabel("Disagreement rate")
    plt.xlabel("Dataset")
    plt.title("Aggregate vs per-source safety disagreement")
    plt.xticks(rotation=30, ha="right")
    save_figure("plot_safety_disagreement", out_dir)

# 📌 Main paper or early appendix


# ================================
# A6. Category agreement / entropy
# ================================
# Plot: entropy vs agreement (scatter)

def plot_category_agreement(category_agree_ds, out_dir):
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        data=category_agree_ds,
        x="avg_entropy",
        y="perfect_agreement_rate",
        size="avg_unique_categories",
        sizes=(50,300),
        legend=True
    )
    plt.xlabel("Average category entropy")
    plt.ylabel("Perfect agreement rate")
    plt.title("Cross-model agreement on AEGIS categories")
    save_figure("plot_category_agreement", out_dir)
# 📌 Main paper 

# ================================
# A7. Confusion vs aggregate label
# ================================
# Plot: heatmap (per model)

def plot_confusion_matrix(confusion_df, model_name, out_dir):
    plt.figure(figsize=(6,5))
    sns.heatmap(
        confusion_df,
        annot=False,
        cmap="Blues",
        square=True
    )
    plt.xlabel("Per-source AEGIS")
    plt.ylabel("Aggregated AEGIS")
    plt.title(f"AEGIS confusion vs aggregate ({model_name})")
    save_figure("plot_confusion_matrix", out_dir)
# 📌 Appendix (one per major model)

# ================================
# A8. Severity-weighted risk profile
# ================================
# Plot: bar plot (severity-aware)

def plot_severity_weighted_risk(sev_table, out_dir):
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=sev_table,
        x="dataset_file",
        y="mean_weighted_harm",
        color="purple"
    )
    plt.ylabel("Severity-weighted harm")
    plt.xlabel("Dataset")
    plt.title("Severity-weighted safety risk")
    plt.xticks(rotation=30, ha="right")
    save_figure("plot_severity_weighted_risk", out_dir)
# 📌 Main paper 


# ================================
# A9. Taxonomy coverage expansion
# ================================
# Plot: entropy distribution
def plot_taxonomy_expansion(expansion_summary, out_dir):
    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=expansion_summary,
        x="dataset_file",
        y="entropy"
    )
    plt.ylabel("AEGIS entropy within source label")
    plt.xlabel("Dataset")
    plt.title("Taxonomy expansion beyond source labels")
    plt.xticks(rotation=30, ha="right")
    save_figure("plot_taxonomy_expansion", out_dir)
# 📌 Main paper (taxonomy contribution)

# ================================
# A10. Reason consistency
# ================================
# Plot: bar plot
def plot_reason_consistency(reason_ds, out_dir):
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=reason_ds,
        x="dataset_file",
        y="avg_unique_reasons",
        color="teal"
    )
    plt.ylabel("Avg. unique reasons per prompt")
    plt.xlabel("Dataset")
    plt.title("Explainability consistency (AEGIS reasons)")
    plt.xticks(rotation=30, ha="right")
    save_figure("plot_reason_consistency", out_dir)
# 📌 Appendix (strong qualitative support)

# ================================
# A11. Language / verbosity bias
# ================================
# Plot: response length comparison
def plot_response_length(src_len, out_dir):
    plt.figure(figsize=(7,4))
    sns.barplot(
        data=src_len,
        x="source_model",
        y="bn_resp_len",
        hue="dataset_file"
    )
    plt.ylabel("Avg Bangla response length")
    plt.xlabel("Model")
    plt.title("Verbosity bias across models")
    plt.xticks(rotation=30, ha="right")
    plt.legend(bbox_to_anchor=(1.05,1))
    save_figure("plot_response_length", out_dir)
# 📌 Appendix (utility vs safety discussion)

# ================================
# A12. Harm-score threshold tradeoff
# ================================
# Plot: tradeoff curve
def plot_threshold_tradeoff(tradeoff_table, out_dir):
    plt.figure(figsize=(6,4))
    plt.plot(
        tradeoff_table["threshold"],
        tradeoff_table["tpr"],
        marker="o",
        label="TPR (unsafe detection)"
    )
    plt.plot(
        tradeoff_table["threshold"],
        tradeoff_table["coverage"],
        marker="s",
        label="Coverage"
    )
    plt.xlabel("Harm score threshold")
    plt.ylabel("Rate")
    plt.title("Threshold tradeoff analysis")
    plt.legend()
    plt.grid(True)
    save_figure("plot_threshold_tradeoff", out_dir)
# 📌 Appendix or main (calibration story)


def save_figure(fig_name, out_dir="figures", dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{fig_name}.png"), dpi=dpi)
    plt.savefig(os.path.join(out_dir, f"{fig_name}.pdf"))
    plt.close()

def plot_all(analyses, df_agg, df_src):
    base_dir = "/home/malam10/projects/ai-safety-bangla/latest/plots"
    plot_dataset_coverage(analyses["A1_completeness_coverage"], f"{base_dir}/main")
    plot_selected_model_distribution(analyses["A2_selected_model_dist"], f"{base_dir}/main")
    plot_agg_harm_distribution(df_agg, f"{base_dir}/main")
    plot_agg_category_distribution(analyses["A4_agg_category_top"], f"{base_dir}/main")
    # plot_safety_disagreement(analyses["A5_safety_disagreement"], f"{base_dir}/main")
    # plot_category_agreement(analyses["A6_category_agreement_ds"], f"{base_dir}/main")
    # plot_severity_weighted_risk(analyses["A8_severity_weighted_agg"], f"{base_dir}/main")
    # plot_taxonomy_expansion(analyses["A9_expansion_summary"], f"{base_dir}/main")
    # plot_reason_consistency(analyses["A10_reason_consistency_ds"], f"{base_dir}/appendix")
    # plot_response_length(analyses["A11_src_len_summary"], f"{base_dir}/appendix")
    # plot_threshold_tradeoff(analyses["A12_threshold_tradeoff"], f"{base_dir}/appendix")


# Pick 6–7 figures:
# Selected model distribution
# Harm score distribution
# AEGIS category distribution
# Category agreement (entropy vs agreement)
# Severity-weighted risk
# Taxonomy expansion
# Threshold tradeoff (optional)
# Everything else → appendix.


