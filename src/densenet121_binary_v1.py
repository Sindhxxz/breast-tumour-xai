# =========================
# DenseNet121 Binary Classification (Colab-ready)
# =========================

from pathlib import Path
import os, time, json
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# =========================
# Config
# =========================
DATA_ROOT = Path("data/preprocessed")  
OUT_DIR   = Path("runs/densenet121_binary"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "densenet121"
FREEZE_BACKBONE = True
UNFREEZE_AT_EPOCH = 3
USE_SAMPLER = True

EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SEED = 42
PATIENCE = 8  # early stopping patience

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
# Dataset
# -------------------------
class BinaryDataset(Dataset):
    """Binary classification: benign vs malignant"""
    def __init__(self, split_root: Path, transform=None):
        self.transform = transform
        self.samples, self.targets = [], []
        for idx, cls in enumerate(["benign", "malignant"]):
            cls_dir = split_root / cls
            if not cls_dir.exists():
                continue
            for st_dir in cls_dir.iterdir():
                if not st_dir.is_dir():
                    continue
                for root, _, files in os.walk(st_dir):
                    for fn in files:
                        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                            self.samples.append(Path(root) / fn)
                            self.targets.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]
        y = self.targets[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, y

# -------------------------
# Transforms
# -------------------------
def make_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE[0], scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

# -------------------------
# Model
# -------------------------
def build_model(num_classes: int = 2):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_f = model.classifier.in_features
    model.classifier = nn.Linear(in_f, num_classes)
    return model.to(DEVICE)

def unfreeze_progressively(model, stage: int):
    """0=classifier only, 1=last dense block, 2=whole network"""
    for p in model.parameters():
        p.requires_grad = False
    if stage >= 0:
        for p in model.classifier.parameters():
            p.requires_grad = True
    if stage >= 1:
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
    if stage >= 2:
        for p in model.parameters():
            p.requires_grad = True

# -------------------------
# Training / Evaluation
# -------------------------
def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train(train)
    losses = []
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1)
        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(np.mean(losses)), acc, f1

def evaluate_test(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["benign","malignant"], digits=4)
    return acc, f1, cm, report

# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)
    train_tfms, eval_tfms = make_transforms()
    
    # Datasets
    train_ds = BinaryDataset(DATA_ROOT / "train", transform=train_tfms)
    val_ds   = BinaryDataset(DATA_ROOT / "val", transform=eval_tfms)
    test_ds  = BinaryDataset(DATA_ROOT / "test", transform=eval_tfms)
    
    # Sampler for imbalanced data
    targets_np = np.array(train_ds.targets)
    if USE_SAMPLER:
        class_counts = np.bincount(targets_np, minlength=2)
        per_class_w = (1.0 / np.clip(class_counts, 1, None)) ** 0.5
        sample_w = per_class_w[targets_np]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_w),
            num_samples=len(sample_w),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = build_model()
    unfreeze_stage = 0
    unfreeze_progressively(model, unfreeze_stage)
    
    # Optimizer & criterion
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0.0
    no_improve = 0
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        # Progressive unfreezing
        if epoch == UNFREEZE_AT_EPOCH:
            unfreeze_stage = 1
            unfreeze_progressively(model, unfreeze_stage)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR/3)
            print(f"[INFO] Unfrozen last block; set lr -> {LR/3}")
        
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, train=False)
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} | {time.time()-t0:.1f}s")
        
        # Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            no_improve = 0
            print(f"[INFO] New best val F1: {best_val_f1:.4f}. Saved checkpoint.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("[INFO] Early stopping.")
                break
    
    # -------------------------
    # Test evaluation
    # -------------------------
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt"))
    acc, f1, cm, report = evaluate_test(model, test_loader)

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
    print("Confusion matrix:", cm)
    print("\nClassification Report:\n", report)
    print(f"[INFO] Results saved to {OUT_DIR / 'test_results.json'}")


if __name__ == "__main__":
    main()
