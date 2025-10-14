# train_resnet50_binary.py
from pathlib import Path
import os, time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# =========================
# Config
# =========================
DATA_ROOT = Path("data/preprocessed")   # train/val/test with benign & malignant folders
OUT_DIR   = Path("runs/resnet50_binary"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "resnet50"
FREEZE_BACKBONE = True
UNFREEZE_AT_EPOCH = 3
USE_SAMPLER = True
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_EPS = 0.05

EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEED = 42
PATIENCE = 8  # early stopping

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def discover_classes(split_root: Path):
    classes = sorted([d.name for d in split_root.iterdir() if d.is_dir()])
    return classes

def collect_split(split_root: Path, classes):
    class_to_idx = {c:i for i,c in enumerate(classes)}
    paths, labels = [], []
    for cls in classes:
        for root, _, files in os.walk(split_root/cls):
            for f in files:
                if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    paths.append(Path(root)/f)
                    labels.append(class_to_idx[cls])
    return paths, labels

class BinaryDataset(Dataset):
    def __init__(self, split_root: Path, classes, transform=None):
        self.transform = transform
        self.samples, self.targets = collect_split(split_root, classes)
        self.classes = classes

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.targets[idx]

# =========================
# Transforms
# =========================
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE[0], scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
eval_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# =========================
# Model
# =========================
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def set_backbone_trainable(model, flag=True):
    for p in model.parameters(): p.requires_grad = flag
    for p in model.fc.parameters(): p.requires_grad = True

# =========================
# Training / Eval
# =========================
def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train(train)
    losses, y_true, y_pred = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train: optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1)
        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
    return np.mean(losses), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def evaluate(model, loader, classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    cm  = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    return acc, f1, cm, report

# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    classes = discover_classes(DATA_ROOT/"train")
    print(f"[INFO] Discovered classes: {classes}")

    train_ds = BinaryDataset(DATA_ROOT/"train", classes, transform=train_tfms)
    val_ds   = BinaryDataset(DATA_ROOT/"val", classes, transform=eval_tfms)
    test_ds  = BinaryDataset(DATA_ROOT/"test", classes, transform=eval_tfms)

    targets_np = np.array(train_ds.targets)
    class_counts = np.bincount(targets_np, minlength=len(classes))
    print("[INFO] Train per-class counts:", dict(zip(classes, class_counts.tolist())))

    # Sampler for imbalance
    if USE_SAMPLER:
        weights = (1.0 / np.clip(class_counts,1,None)) ** 0.5
        sample_w = weights[targets_np]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_w),
                                        num_samples=len(sample_w),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Loss & Model
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING_EPS if USE_LABEL_SMOOTHING else 0.0).to(DEVICE)
    model = build_model(num_classes=len(classes)).to(DEVICE)
    if FREEZE_BACKBONE: set_backbone_trainable(model, False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Resume if checkpoint exists
    start_epoch = 1
    best_val_f1 = -1
    ckpt_path = OUT_DIR / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        print(f"[INFO] Resuming from epoch {start_epoch}")

    no_improve = 0

    # =========================
    # Training loop
    # =========================
    for epoch in range(start_epoch, EPOCHS+1):
        t0 = time.time()
        # Unfreeze last block
        if epoch == UNFREEZE_AT_EPOCH:
            for p in model.layer4.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR/3)
            print(f"[INFO] Unfrozen last block; set lr -> {LR/3}")

        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, train=False)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} | lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")

        # Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "classes": classes
            }, ckpt_path)
            print(f"[INFO] New best val F1: {best_val_f1:.4f}. Saved checkpoint.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("[INFO] Early stopping.")
                break

    # =========================
    # Test evaluation
    # =========================
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    acc, f1, cm, report = evaluate(model, test_loader, classes)
    # Save results in JSON
    results_dict = {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }

    import json
    with open(OUT_DIR / "test_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print("\n=== TEST METRICS ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()
