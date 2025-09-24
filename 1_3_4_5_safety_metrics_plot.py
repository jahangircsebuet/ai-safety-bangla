import matplotlib.pyplot as plt

def plot_safety_comparison(baseline_metrics: dict, finetuned_metrics: dict, save_path="safety_comparison.png"):
    """
    Plot side-by-side comparison of safety metrics for baseline vs fine-tuned model.

    Args:
        baseline_metrics (dict): Metrics from baseline model
        finetuned_metrics (dict): Metrics from fine-tuned model
        save_path (str): Where to save the figure
    """

    # Only keep the 4 key metrics (ignore counts)
    metric_keys = [
        "rr_rate",
        "ur_rate",
        "other_rate",
        "over_refusal_rate"
    ]

    baseline_vals = [baseline_metrics.get(k, 0) for k in metric_keys]
    finetuned_vals = [finetuned_metrics.get(k, 0) for k in metric_keys]

    x = range(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], baseline_vals, width, label="Baseline", color="#d9534f")
    ax.bar([i + width/2 for i in x], finetuned_vals, width, label="Fine-tuned", color="#5cb85c")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Safety Metrics: Baseline vs Fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=20, ha="right")
    ax.legend()

    # Annotate bars
    for i, v in enumerate(baseline_vals):
        ax.text(i - width/2, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    for i, v in enumerate(finetuned_vals):
        ax.text(i + width/2, v + 1, f"{v:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Safety comparison plot saved to {save_path}")
    plt.close()




def plot_multi_model_comparison(metrics_dict: dict, save_path="multi_model_comparison.png"):
    """
    Plot side-by-side comparison of safety metrics for multiple models.

    Args:
        metrics_dict (dict): {model_name: metrics_dict}
        save_path (str): Where to save the figure
    """

    # Only keep the 4 key metrics (ignore counts)
    metric_keys = [
        "rr_rate",
        "ur_rate",
        "other_rate",
        "over_refusal_rate"
    ]

    model_names = list(metrics_dict.keys())
    n_models = len(model_names)

    # Collect values for each model
    values = {m: [metrics_dict[m].get(k, 0) for k in metric_keys] for m in model_names}

    x = range(len(metric_keys))
    width = 0.8 / n_models  # adjust bar width to fit all models

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        offsets = [xi - 0.4 + (i+0.5)*width for xi in x]
        ax.bar(offsets, values[model_name], width, label=model_name)

        # Annotate bars
        for xi, v in zip(offsets, values[model_name]):
            ax.text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=7)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Safety Metrics Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=20, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Multi-model comparison plot saved to {save_path}")
    plt.close()
