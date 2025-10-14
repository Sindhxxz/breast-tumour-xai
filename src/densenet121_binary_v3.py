# ==============================
# DenseNet121 Binary Classification (Improved)
# ==============================

from pathlib import Path
import time, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image

# --------------------
# Config
# --------------------
DATA_ROOT = Path("/content/drive/MyDrive/fibroadenoma/preprocessed")
OUT_DIR   = Path("runs/densenet121_improved"); OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 2
FREEZE_BACKBONE = True
EPOCHS = 30
BATCH_SIZE = 16
BASE_LR = 1e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --------------------
# Dataset
# --------------------
class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples, self.targets = [], []
        for idx, cls in enumerate(["benign", "malignant"]):
            cls_dir = Path(root_dir) / cls
            for img_file in cls_dir.rglob("*.*"):
                if img_file.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff",".bmp"]:
                    self.samples.append(img_file)
                    self.targets.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label

train_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = BinaryDataset(DATA_ROOT / "train", transform=train_tfms)
val_ds   = BinaryDataset(DATA_ROOT / "val",   transform=eval_tfms)
test_ds  = BinaryDataset(DATA_ROOT / "test",  transform=eval_tfms)

# Weighted sampler for class imbalance
train_targets = np.array(train_ds.targets)
class_counts  = np.bincount(train_targets, minlength=NUM_CLASSES)
sample_weights = 1. / class_counts[train_targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Class weights for loss
class_weights = class_counts.sum() / (NUM_CLASSES * np.clip(class_counts, 1, None))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# --------------------
# Model
# --------------------
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)

if FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)


# --------------------
# Checkpoint resume
# --------------------
ckpt_path = OUT_DIR / "checkpoint.pth"
start_epoch = 1
best_metric = 0.0

if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_metric = ckpt.get("best_metric", 0.0)
    print(f"[INFO] Resuming from epoch {start_epoch}")

# --------------------
# Training / Evaluation
# --------------------
def run_epoch(loader, train=True):
    model.train(train)
    losses, all_y, all_p = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1)
        all_y.extend(y.detach().cpu().numpy())
        all_p.extend(preds.detach().cpu().numpy())
    acc = accuracy_score(all_y, all_p)
    f1  = f1_score(all_y, all_p, average="macro")
    return float(np.mean(losses)), acc, f1

patience, no_improve = 5, 0

for epoch in range(start_epoch, EPOCHS+1):
    t0 = time.time()
    train_loss, train_acc, train_f1 = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_f1 = run_epoch(val_loader, train=False)
    combined_metric = 0.5*val_acc + 0.5*val_f1
    scheduler.step(combined_metric)

    print(f"Epoch {epoch:02d} | "
          f"train_loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
          f"val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")

    if combined_metric > best_metric + 1e-4:
        best_metric = combined_metric
        no_improve = 0
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric
        }, ckpt_path)
        torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
        print(f"[INFO] New best combined metric: {best_metric:.4f}. Saved checkpoint.")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("[INFO] Early stopping.")
            break

# --------------------
# Final Test Evaluation
# --------------------
model.load_state_dict(torch.load(OUT_DIR / "best_model.pth", map_location=DEVICE))
model.eval()

all_y, all_p = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1).cpu().numpy()
        all_p.extend(preds)
        all_y.extend(y.numpy())

acc = accuracy_score(all_y, all_p)
f1  = f1_score(all_y, all_p, average="macro")
cm  = confusion_matrix(all_y, all_p).tolist()
report = classification_report(all_y, all_p, target_names=["benign","malignant"], digits=4)

print("\n=== TEST METRICS (Binary) ===")
print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
print("Confusion matrix:", cm)
print(report)

(Path(OUT_DIR) / "metrics.json").write_text(json.dumps(
    {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "report": report}, indent=2))