import json
import pandas as pd
from safety_metrics import compute_safety_metrics
from 1_3_4_5_safety_metrics_plot import plot_multi_model_comparison

def evaluate_outputs(outputs_dict, save_plot="safety_comparison.png"):
    metrics_dict = {}
    for model_name, file_path in outputs_dict.items():
        df = pd.read_json(file_path)
        metrics = compute_safety_metrics(df,
                                         prompt_col="prompt_bn",
                                         response_col="model_response",
                                         safety_col="prompt_safety")
        metrics_dict[model_name] = metrics

    plot_multi_model_comparison(metrics_dict, save_path=save_plot)

    print("📊 Metrics by model:")
    for model, m in metrics_dict.items():
        print(model, ":", m)

    with open("metrics_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

    return metrics_dict
