import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
 
FORCE_LABELS = ["Pos1", "Pos2", "Pos3"]
OUT_DIR = "grafikni"
os.makedirs(OUT_DIR, exist_ok=True)
 
 
def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл {path} — сначала запустите скрипт, который его создаёт")
    data = np.load(path)
    return data["y_true"], data["y_pred"]
 
 
RESULTS = {
    "kNN (LOSO)": load_npz("knn_loso_results.npz"),
    "RF (LOSO)": load_npz("rf_loso_results.npz"),
    "CNN (LOSO)": load_npz("cnn_loso_results.npz"),
}
 
 
def plot_predicted_vs_actual(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.4)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual force (N)")
        ax.set_ylabel("Predicted force (N)")
        ax.set_title(FORCE_LABELS[i])
    fig.suptitle(f"{model_name}: Predicted vs Actual (cross-session)")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUT_DIR}/pred_vs_actual_{safe_name}.png", dpi=200)
    plt.close()
 
 
rows = []
for name, (y_true, y_pred) in RESULTS.items():
    plot_predicted_vs_actual(y_true, y_pred, name)
    rows.append({"model": name,
                 "MAE": mean_absolute_error(y_true, y_pred),
                 "R2": r2_score(y_true, y_pred)})
 
summary = pd.DataFrame(rows)
print(summary.round(4).to_string(index=False))
summary.to_csv(f"{OUT_DIR}/loso_summary.csv", index=False)
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(summary["model"], summary["MAE"], color="indianred")
axes[0].set_ylabel("Mean MAE (N)")
axes[0].set_title("LOSO: Mean Absolute Error")
axes[0].tick_params(axis="x", rotation=15)
 
axes[1].bar(summary["model"], summary["R2"], color="steelblue")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Mean R2")
axes[1].set_title("LOSO: Coefficient of Determination")
axes[1].tick_params(axis="x", rotation=15)
 
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/loso_comparison.png", dpi=200)
plt.close()