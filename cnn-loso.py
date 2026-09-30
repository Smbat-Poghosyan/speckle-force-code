import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
 
CSV_PATH = "labels.csv"
IMG_DIR = "nkarner"
FORCE_COLS = ["force_p1_N", "force_p2_N", "force_p3_N"]
SETS = ["A", "B", "C", "D", "E"]
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
df = pd.read_csv(CSV_PATH)
 
 
class SpeckleDataset(Dataset):
    def __init__(self, frame_df):
        self.frame_df = frame_df.reset_index(drop=True)
 
    def __len__(self):
        return len(self.frame_df)
 
    def __getitem__(self, idx):
        row = self.frame_df.iloc[idx]
        img = Image.open(os.path.join(IMG_DIR, row["filename"])).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        labels = torch.tensor(row[FORCE_COLS].values.astype(np.float32))
        return tensor, labels
 
 
class SpeckleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
 
    def forward(self, x):
        x = self.features(x)
        return self.head(x)
 
 
def train_one_fold(train_df, test_df):
    train_loader = DataLoader(SpeckleDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(SpeckleDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
    model = SpeckleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
 
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        running_loss /= len(train_loader.dataset)
        print(f"    epoch {epoch + 1}/{EPOCHS} - train_loss={running_loss:.4f}")
 
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)
 
 
results = []
cnn_true_all, cnn_pred_all = [], []
for held_out in SETS:
    print(f"Fold: held-out set = {held_out}")
    train_df = df[df["set"] != held_out]
    test_df = df[df["set"] == held_out]
 
    y_true, y_pred = train_one_fold(train_df, test_df)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append((held_out, mae, r2))
    cnn_true_all.append(y_true)
    cnn_pred_all.append(y_pred)
    print(f"  Held-out {held_out}: MAE={mae:.4f} N, R2={r2:.4f}")
 
maes = [m for _, m, _ in results]
r2s = [r for _, _, r in results]
print(f"CNN LOSO mean: MAE={np.mean(maes):.4f}, R2={np.mean(r2s):.4f}")
 
np.savez("cnn_loso_results.npz", y_true=np.concatenate(cnn_true_all), y_pred=np.concatenate(cnn_pred_all))