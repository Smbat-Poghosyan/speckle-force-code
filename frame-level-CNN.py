import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
 
CSV_PATH = "labels.csv"
IMG_DIR = "nkarner"
FORCE_COLS = ["force_p1_N", "force_p2_N", "force_p3_N"]
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
df = pd.read_csv(CSV_PATH)
 
records = df.drop_duplicates(["set", "record_id"])[["set", "record_id"]]
train_val, test = train_test_split(records, test_size=83 / 550, random_state=42)
train, val = train_test_split(train_val, test_size=83 / len(train_val), random_state=42)
 
 
def select_frames(split_records):
    keys = set(zip(split_records["set"], split_records["record_id"]))
    mask = df.apply(lambda r: (r["set"], r["record_id"]) in keys, axis=1)
    return df[mask].reset_index(drop=True)
 
 
train_df = select_frames(train)
val_df = select_frames(val)
test_df = select_frames(test)
print("Train:", len(train_df), " Val:", len(val_df), " Test:", len(test_df))
 
 
class SpeckleDataset(Dataset):
    def __init__(self, frame_df):
        self.frame_df = frame_df
 
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
 
 
train_loader = DataLoader(SpeckleDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(SpeckleDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(SpeckleDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
 
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
 
 
model = SpeckleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
 
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)
 
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            val_loss += loss_fn(preds, labels).item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
 
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).cpu().numpy()
        all_preds.append(preds)
        all_true.append(labels.numpy())
all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)
 
mae = mean_absolute_error(all_true, all_preds)
r2 = r2_score(all_true, all_preds)
print(f"CNN frame-level test: MAE={mae:.4f} N, R2={r2:.4f}")
for i, pos in enumerate(["Pos1", "Pos2", "Pos3"]):
    m = mean_absolute_error(all_true[:, i], all_preds[:, i])
    r = r2_score(all_true[:, i], all_preds[:, i])
    print(f"  {pos}: MAE={m:.4f}, R2={r:.4f}")