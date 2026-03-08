import os
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from torchvision import transforms
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
DATA_DIR = "train_data/ff++_datacrop"
SAVE_PATH = "models/best_model.pth"

IMG_SIZE = 380
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# =========================
# Seed
# =========================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# =========================
# Dataset
# =========================
class FFPPFrameDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# =========================
# Transforms
# =========================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Load Data
# =========================
real_imgs = glob.glob(os.path.join(DATA_DIR, "real", "*.jpg"))
fake_imgs = glob.glob(os.path.join(DATA_DIR, "fake", "*.jpg"))

paths = real_imgs + fake_imgs
labels = [0] * len(real_imgs) + [1] * len(fake_imgs)

X_train, X_val, y_train, y_val = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

train_ds = FFPPFrameDataset(X_train, y_train, train_tf)
val_ds = FFPPFrameDataset(X_val, y_val, val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

# =========================
# Model
# =========================
model = timm.create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=1
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# Train
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x.to(DEVICE)
        y = torch.tensor(y).float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # =========================
    # Validation
    # =========================
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = torch.tensor(y).to(DEVICE)

            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze()

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("🔥 Best model saved!")

print("Training finished.")