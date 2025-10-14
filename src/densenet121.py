# src/train_densenet121_fixed.py
from __future__ import annotations
import json, time, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# --------------------
# Config
# --------------------
DATA_ROOT = Path("data/preprocessed")
OUT_DIR   = Path("runs/densenet121_fixed"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "densenet121"
SEED = 42
FREEZE_BACKBONE = True         # warmup: train head first
UNFREEZE_STAGE1_EPOCH = 3     # then unfreeze last block
UNFREEZE_STAGE2_EPOCH = 8     # then unfreeze whole net
EPOCHS = 30
BATCH_SIZE = 32
BASE_LR = 1e-4
GRAD_CLIP = 2.0
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
PATIENCE = 8

# --------------------
# Reproducibility
# --------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------
# Dataset helpers
# --------------------
def discover_subtypes(train_root: Path) -> List[str]:
    subtypes = set()
    for cat in ("benign","malignant"):
        d = train_root / cat
        if not d.exists(): continue
        for st in d.iterdir():
            if st.is_dir(): subtypes.add(st.name)
    subtypes = sorted(subtypes)
    print(f"[INFO] Discovered subtypes ({len(subtypes)}): {subtypes}")
    return subtypes

def collect_split(split_root: Path, subtypes: List[str]) -> Tuple[List[Path], List[int]]:
    st2idx = {st:i for i,st in enumerate(subtypes)}
    img_paths, labels = [], []
    for cat in ("benign","malignant"):
        cat_dir = split_root / cat
        if not cat_dir.exists(): continue
        for st_dir in cat_dir.iterdir():
            if not st_dir.is_dir(): continue
            st = st_dir.name
            if st not in st2idx: continue
            for root, _, files in os.walk(st_dir):
                for fn in files:
                    if fn.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff",".bmp")):
                        img_paths.append(Path(root)/fn)
                        labels.append(st2idx[st])
    return img_paths, labels

class SubtypeDataset(Dataset):
    def __init__(self, split_root: Path, subtypes: List[str], transform=None):
        self.transform = transform
        self.subtypes = subtypes
        self.samples, self.targets = collect_split(split_root, subtypes)
        assert len(self.samples) == len(self.targets)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p = self.samples[idx]
        y = self.targets[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, y

# --------------------
# Transforms (gentle for histopathology)
# --------------------
def make_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

# --------------------
# Model
# --------------------
def build_model(name: str, num_classes: int) -> nn.Module:
    if name.lower() == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
    elif name.lower() == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    else:
        raise ValueError("Unsupported model")
    return m

# progressive unfreeze (no optimizer recreation)
def unfreeze_progressively(model: nn.Module, model_name: str, stage: int):
    # stage: 0 = head only, 1 = head + last block, 2 = all
    for p in model.parameters():
        p.requires_grad = False
    if model_name == "densenet121":
        if stage >= 0:
            for p in model.classifier.parameters(): p.requires_grad = True
        if stage >= 1:
            for p in model.features.denseblock4.parameters(): p.requires_grad = True
        if stage >= 2:
            for p in model.parameters(): p.requires_grad = True
    elif model_name == "resnet50":
        if stage >= 0:
            for p in model.fc.parameters(): p.requires_grad = True
        if stage >= 1:
            for p in model.layer4.parameters(): p.requires_grad = True
        if stage >= 2:
            for p in model.parameters(): p.requires_grad = True

def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g['lr'] = lr

def add_new_params_to_optimizer(optimizer, model, already_in_optimizer, lr):
    new_params = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in already_in_optimizer:
            new_params.append(p)
            already_in_optimizer.add(id(p))
    if new_params:
        optimizer.add_param_group({'params': new_params, 'lr': lr})


# --------------------
# Training / Eval helpers
# --------------------
def run_epoch(model, loader, criterion, optimizer, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
    f1  = f1_score(y_true, y_pred, average="macro") if len(y_true)>0 else 0.0
    return float(np.mean(losses)), acc, f1

def evaluate_test(model, loader, classes: List[str]):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    cm  = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    return acc, f1, cm, report

# --------------------
# Main
# --------------------
def main():
    set_seed(SEED)

    # transforms & datasets
    train_tfms, eval_tfms = make_transforms()
    SUBTYPES = discover_subtypes(DATA_ROOT / "train")
    train_ds = SubtypeDataset(DATA_ROOT / "train", SUBTYPES, transform=train_tfms)
    val_ds   = SubtypeDataset(DATA_ROOT / "val",   SUBTYPES, transform=eval_tfms)
    test_ds  = SubtypeDataset(DATA_ROOT / "test",  SUBTYPES, transform=eval_tfms)

    # class counts and class weights for CE
    targets_np = np.array(train_ds.targets)
    class_counts = np.bincount(targets_np, minlength=len(SUBTYPES))
    print("[INFO] Train per-class counts:", dict(zip(SUBTYPES, class_counts.tolist())))
    # compute simple inverse-frequency weights (normalized)
    weights = class_counts.sum() / (len(class_counts) * np.clip(class_counts, 1, None))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    # dataloaders (no sampler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # model
    model = build_model(MODEL_NAME, num_classes=len(SUBTYPES)).to(DEVICE)

    # progressive freeze: start head only
    unfreeze_stage = 0
    unfreeze_progressively(model, MODEL_NAME, unfreeze_stage)
    if FREEZE_BACKBONE:
        # ensure only head trainable (done above); optimizer will only include those params
        pass

    # optimizer (only parameters with requires_grad=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    no_improve = 0

    history = {"train_loss":[], "train_acc":[], "train_f1":[],
               "val_loss":[], "val_acc":[], "val_f1":[]}

    # track which params are already in optimizer
    already_in_optimizer = set(id(p) for g in optimizer.param_groups for p in g['params'])

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()

        if epoch == UNFREEZE_STAGE1_EPOCH:
            unfreeze_stage = 1
            unfreeze_progressively(model, MODEL_NAME, unfreeze_stage)
            add_new_params_to_optimizer(optimizer, model, already_in_optimizer, BASE_LR/3)
            set_lr(optimizer, BASE_LR/3)
            print("[INFO] Unfrozen last block; set lr ->", BASE_LR/3)

        if epoch == UNFREEZE_STAGE2_EPOCH:
            unfreeze_stage = 2
            unfreeze_progressively(model, MODEL_NAME, unfreeze_stage)
            add_new_params_to_optimizer(optimizer, model, already_in_optimizer, BASE_LR/5)
            set_lr(optimizer, BASE_LR/5)
            print("[INFO] Unfrozen full network; set lr ->", BASE_LR/5)


        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss,   val_acc,   val_f1   = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc); history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} | lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")

        # Save best by VAL macro-F1
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "subtypes": SUBTYPES,
                "model_name": MODEL_NAME,
            }, OUT_DIR / "best.pt")
            print(f"[INFO] New best val F1: {best_val_f1:.4f}. Saved checkpoint.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("[INFO] Early stopping.")
                break

    # final evaluation
    ckpt = torch.load(OUT_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    acc, f1, cm, report = evaluate_test(model, test_loader, SUBTYPES)

    print("\n=== TEST METRICS (8-class) ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print("Confusion matrix (8x8):\n", cm)
    print("\nClassification Report:\n", report)

    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "report": report
    }, indent=2))
    print("[INFO] Saved:", OUT_DIR / "best.pt", OUT_DIR / "metrics.json")

if __name__ == "__main__":
    # multiprocessing guard for Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()
