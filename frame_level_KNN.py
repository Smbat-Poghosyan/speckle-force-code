import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
 
CSV_PATH = "labels.csv"
IMG_DIR = "nkarner"
FORCE_COLS = ["force_p1_N", "force_p2_N", "force_p3_N"]
 
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
    return np.array(stats16 + speckle3)
 
 
records = df.drop_duplicates(["set", "record_id"])[["set", "record_id"]]
train_val, test = train_test_split(records, test_size=83 / 550, random_state=42)
train, val = train_test_split(train_val, test_size=83 / len(train_val), random_state=42)
 
 
def build_dataset(split_records):
    keys = set(zip(split_records["set"], split_records["record_id"]))
    rows = df[df.apply(lambda r: (r["set"], r["record_id"]) in keys, axis=1)]
    X, y = [], []
    for _, row in rows.iterrows():
        img = Image.open(os.path.join(IMG_DIR, row["filename"])).convert("L")
        X.append(frame_features(img))
        y.append(row[FORCE_COLS].values)
    return np.array(X), np.array(y)
 
 
X_train, y_train = build_dataset(train)
X_test, y_test = build_dataset(test)
print("Train frames:", len(X_train), " Test frames:", len(X_test))
 
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance")),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
 
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"kNN frame-level: MAE={mae:.4f} N, R2={r2:.4f}")
for i, pos in enumerate(["Pos1", "Pos2", "Pos3"]):
    m = mean_absolute_error(y_test[:, i], pred[:, i])
    r = r2_score(y_test[:, i], pred[:, i])
    print(f"  {pos}: MAE={m:.4f}, R2={r:.4f}")