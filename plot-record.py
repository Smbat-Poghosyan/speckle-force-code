import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
 
FORCE_LABELS = ["Pos1", "Pos2", "Pos3"]
OUT_DIR = "grafikni"
 
 
def load_npz(path):
    data = np.load(path)
    return data["y_true"], data["y_pred"]
 
 
RESULTS = {
    "kNN (record)":  load_npz("knn_record_results.npz"),
    "RF (record)":   load_npz("rf_record_results.npz"),
    "CNN (record)": load_npz("cnn_record_results.npz"),
}
 
 
def plot_predicted_vs_actual(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], s=12, alpha=0.6)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual force (N)")
        ax.set_ylabel("Predicted force (N)")
        ax.set_title(FORCE_LABELS[i])
    fig.suptitle(f"{model_name}: Predicted vs Actual")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT_DIR}/pred_vs_actual_{safe_name}.png", dpi=200)
    plt.close()
 
 
def build_summary_table(results_dict):
    rows = []
    for name, (y_true, y_pred) in results_dict.items():
        row = {"model": name,
               "MAE": mean_absolute_error(y_true, y_pred),
               "R2": r2_score(y_true, y_pred)}
        for i, pos in enumerate(FORCE_LABELS):
            row[f"MAE_{pos}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            row[f"R2_{pos}"] = r2_score(y_true[:, i], y_pred[:, i])
        rows.append(row)
    return pd.DataFrame(rows)
 
 
import os
os.makedirs(OUT_DIR, exist_ok=True)
 
for name, (y_true, y_pred) in RESULTS.items():
    plot_predicted_vs_actual(y_true, y_pred, name)
 
summary = build_summary_table(RESULTS)
print(summary.round(4).to_string(index=False))
summary.to_csv(f"{OUT_DIR}/results_summary.csv", index=False)
 
plt.figure(figsize=(6, 4))
plt.bar(summary["model"], summary["MAE"], color="steelblue")
plt.ylabel("Overall MAE (N)")
plt.title("Model Comparison: Overall MAE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison_mae.png", dpi=200)
plt.close()