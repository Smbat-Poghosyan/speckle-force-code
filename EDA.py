import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

CSV_PATH = "labels.csv"
IMG_DIR = "nkarner"               # optional folder with the .png files
OUT_DIR = "grafikni"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
force_cols = ["force_p1_N", "force_p2_N", "force_p3_N"]
print("Shape:", df.shape)
print(df.describe())

# 1) Records per acquisition set
counts = df.groupby("set")["record_id"].nunique()
plt.figure(figsize=(6, 4))
counts.plot(kind="bar", color="steelblue")
plt.title("Number of Records per Acquisition Set")
plt.xlabel("Acquisition Set"); plt.ylabel("Number of Records")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/records_per_set.png", dpi=200); plt.close()

# 2) Force distributions
plt.figure(figsize=(7, 5))
for col, label in zip(force_cols, ["Pos1", "Pos2", "Pos3"]):
    sns.histplot(df[col], bins=40, label=label, kde=True, alpha=0.4)
plt.title("Distribution of Applied Forces by Sensing Position")
plt.xlabel("Force (N)"); plt.ylabel("Count"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/force_histograms.png", dpi=200); plt.close()

# 3) Boxplot per position
plt.figure(figsize=(6, 5))
sns.boxplot(data=df[force_cols])
plt.xticks([0, 1, 2], ["Pos1", "Pos2", "Pos3"]); plt.ylabel("Force (N)")
plt.title("Force Magnitude Distribution per Sensing Position")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/force_boxplot.png", dpi=200); plt.close()

# 4) Record-level forces sorted by total force
rec = df.drop_duplicates(subset=["set", "record_id"])[force_cols].copy()
rec["total"] = rec.sum(axis=1)
rec = rec.sort_values("total").reset_index(drop=True)
plt.figure(figsize=(7, 5))
for col, label in zip(force_cols, ["Pos1", "Pos2", "Pos3"]):
    plt.scatter(rec.index, rec[col], s=8, label=label)
plt.xlabel("Record index (sorted by total force)"); plt.ylabel("Force (N)")
plt.title("Force Labels Across All Records"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/force_by_record.png", dpi=200); plt.close()

# 5) Correlation heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(df[force_cols].corr(), annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Correlation Between Force Channels")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/force_correlation.png", dpi=200); plt.close()

# 6) Force ranges by set and position
melted = df.melt(id_vars="set", value_vars=force_cols,
                  var_name="position", value_name="force")
plt.figure(figsize=(8, 5))
sns.boxplot(data=melted, x="set", y="force", hue="position")
plt.title("Force Ranges by Acquisition Set and Position"); plt.ylabel("Force (N)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/force_by_set.png", dpi=200); plt.close()

# 7) Optional: sample images, only if you download the PNGs locally
if os.path.isdir(IMG_DIR):
    sample_files = df["filename"].sample(8, random_state=0).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, fname in zip(axes.ravel(), sample_files):
        path = os.path.join(IMG_DIR, fname)
        if os.path.exists(path):
            ax.imshow(Image.open(path).convert("L"), cmap="gray")
            r = df[df["filename"] == fname].iloc[0]
            ax.set_title(f"{r['force_p1_N']},{r['force_p2_N']},{r['force_p3_N']} N", fontsize=8)
        ax.axis("off")
    plt.suptitle("Sample Specklegrams with Force Labels")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/sample_images.png", dpi=200); plt.close()
else:
    print(f"No '{IMG_DIR}' folder found — skipping image preview.")

print("All plots saved to:", OUT_DIR)