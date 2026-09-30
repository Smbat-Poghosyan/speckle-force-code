import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
 
CSV_PATH = "labels.csv"
IMG_DIR = "nkarner"
FORCE_COLS = ["force_p1_N", "force_p2_N", "force_p3_N"]
SETS = ["A", "B", "C", "D", "E"]
 
df = pd.read_csv(CSV_PATH)
 
 
def frame_features(img):
    arr = np.asarray(img, dtype=np.float64)
    flat = arr.ravel()
    p10, p25, p50, p75, p90 = np.percentile(flat, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(flat, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    energy_img = np.sum((flat / 255.0) ** 2) / flat.size
    glcm = graycomatrix(arr.astype(np.uint8), [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    speckle_contrast = flat.std() / flat.mean()
    f = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(f) ** 2
    yy, xx = np.indices(power.shape)
    cy, cx = np.array(power.shape) // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    centroid = np.sum(r * power) / np.sum(power)
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(arr - arr.mean())) ** 2).real
    autocorr = np.fft.fftshift(autocorr)
    autocorr /= autocorr.max()
    profile = autocorr[cy, cx:]
    below = np.where(profile < 0.5)[0]
    grain_radius = below[0] if len(below) > 0 else len(profile)
    stats16 = [flat.mean(), flat.std(), flat.min(), flat.max(), p10, p25, p50, p75, p90,
               p75 - p25, entropy, energy_img, contrast, homogeneity, glcm_energy, correlation]
    speckle3 = [speckle_contrast, centroid, grain_radius]
    return np.array(stats16), np.array(speckle3)
 
 
def record_features(group):
    group = group.sort_values("frame_idx")
    stats_all, speckle_all, frames = [], [], []
    for fname in group["filename"]:
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("L")
        stats16, speckle3 = frame_features(img)
        stats_all.append(stats16)
        speckle_all.append(speckle3)
        frames.append(np.asarray(img, dtype=np.float64))
    stats_all = np.array(stats_all)
    speckle_all = np.array(speckle_all)
    feat32 = np.concatenate([stats_all.mean(axis=0), stats_all.std(axis=0)])
    feat6 = np.concatenate([speckle_all.mean(axis=0), speckle_all.std(axis=0)])
    ssim_vals = np.array([ssim(frames[i], frames[i + 1], data_range=255) for i in range(len(frames) - 1)])
    slope = np.polyfit(np.arange(len(ssim_vals)), ssim_vals, 1)[0]
    auc = np.trapezoid(ssim_vals) / len(ssim_vals)
    feat_ssim = [ssim_vals.mean(), ssim_vals.std(), ssim_vals.min(), ssim_vals[-1], slope, auc]
    return np.concatenate([feat32, feat6, feat_ssim])
 
 
print("Extracting record-level features for all 550 records...")
records = df.drop_duplicates(["set", "record_id"])[["set", "record_id"]].reset_index(drop=True)
X_all, y_all, set_all = [], [], []
for _, row in records.iterrows():
    group = df[(df["set"] == row["set"]) & (df["record_id"] == row["record_id"])]
    X_all.append(record_features(group))
    y_all.append(group[FORCE_COLS].iloc[0].values)
    set_all.append(row["set"])
X_all = np.array(X_all)
y_all = np.array(y_all)
set_all = np.array(set_all)
 
knn_model = Pipeline([("scaler", StandardScaler()),
                       ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))])
rf_model = RandomForestRegressor(n_estimators=400, max_features="sqrt", random_state=42, n_jobs=-1)
 
results = {"kNN": [], "RF": []}
knn_true_all, knn_pred_all = [], []
rf_true_all, rf_pred_all = [], []
for held_out in SETS:
    train_mask = set_all != held_out
    test_mask = set_all == held_out
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
 
    knn_model.fit(X_train, y_train)
    pred_knn = knn_model.predict(X_test)
    mae_knn = mean_absolute_error(y_test, pred_knn)
    r2_knn = r2_score(y_test, pred_knn)
    results["kNN"].append((held_out, mae_knn, r2_knn))
    knn_true_all.append(y_test)
    knn_pred_all.append(pred_knn)
 
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    r2_rf = r2_score(y_test, pred_rf)
    results["RF"].append((held_out, mae_rf, r2_rf))
    rf_true_all.append(y_test)
    rf_pred_all.append(pred_rf)
 
    print(f"Held-out {held_out}: kNN MAE={mae_knn:.4f} R2={r2_knn:.3f} | "
          f"RF MAE={mae_rf:.4f} R2={r2_rf:.3f}")
 
for model_name in ["kNN", "RF"]:
    maes = [m for _, m, _ in results[model_name]]
    r2s = [r for _, _, r in results[model_name]]
    print(f"{model_name} mean: MAE={np.mean(maes):.4f}, R2={np.mean(r2s):.3f}")
 
np.savez("knn_loso_results.npz", y_true=np.concatenate(knn_true_all), y_pred=np.concatenate(knn_pred_all))
np.savez("rf_loso_results.npz", y_true=np.concatenate(rf_true_all), y_pred=np.concatenate(rf_pred_all))