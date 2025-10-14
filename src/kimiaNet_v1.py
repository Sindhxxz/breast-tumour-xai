# src/train_kimianet.py
from pathlib import Path
import time, json, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image
from typing import List, Tuple

# --------------------
# Config
# --------------------
DATA_ROOT = Path("data/preprocessed")
OUT_DIR   = Path("runs/kimianet"); OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = Path("weights/KimiaNetPyTorchWeights.pth")
NUM_CLASSES = 8

FREEZE_BACKBONE = True
EPOCHS = 30
BATCH_SIZE = 16       # smaller batch size for CPU
BASE_LR = 1e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# --------------------
# Data
# --------------------
def discover_subtypes(train_root: Path) -> List[str]:
    subtypes = set()
    for cat in ["benign", "malignant"]:
        cat_dir = train_root / cat
        if not cat_dir.exists():
            continue
        for d in cat_dir.iterdir():
            if d.is_dir():
                subtypes.add(d.name)
    return sorted(subtypes)

def collect_split(split_root: Path, subtypes: List[str]) -> Tuple[list, list]:
    st2idx = {st: i for i, st in enumerate(subtypes)}
    img_paths, labels = [], []
    for cat in ["benign", "malignant"]:
        cat_dir = split_root / cat
        if not cat_dir.exists():
            continue
        for st_dir in cat_dir.iterdir():
            if not st_dir.is_dir(): 
                continue
            st = st_dir.name
            if st not in st2idx:
                continue
            for root, _, files in os.walk(st_dir):
                for fn in files:
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                        img_paths.append(Path(root) / fn)
                        labels.append(st2idx[st])
    return img_paths, labels

class SubtypeDataset(torch.utils.data.Dataset):
    def __init__(self, split_root: Path, subtypes: List[str], transform=None):
        self.transform = transform
        self.samples, self.targets = collect_split(split_root, subtypes)

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
# Transforms
# --------------------
train_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Discover subtype names
SUBTYPES = discover_subtypes(DATA_ROOT / "train")
print("[INFO] Discovered subtypes:", SUBTYPES)

# Build datasets & loaders
train_ds = SubtypeDataset(DATA_ROOT / "train", SUBTYPES, transform=train_tfms)
val_ds   = SubtypeDataset(DATA_ROOT / "val",   SUBTYPES, transform=eval_tfms)
test_ds  = SubtypeDataset(DATA_ROOT / "test",  SUBTYPES, transform=eval_tfms)

train_targets = np.array(train_ds.targets)
class_counts  = np.bincount(train_targets, minlength=len(SUBTYPES))
class_weights = class_counts.sum() / (len(class_counts) * np.clip(class_counts, 1, None))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader  = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())


# --------------------
# KimiaNet Model
# --------------------
def build_kimianet(weights_path: Path, num_classes: int):
    model = models.densenet121(weights=None)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model.to(DEVICE)

model = build_kimianet(WEIGHTS_PATH, NUM_CLASSES)

if FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


# --------------------
# Resume Training
# --------------------
start_epoch = 1
ckpt_path = OUT_DIR / "checkpoint.pth"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[INFO] Resuming from epoch {start_epoch}")


# --------------------
# Train / Eval Loops
# --------------------
def run_epoch(loader, train: bool):
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


best_val_f1 = 0.0
patience, no_improve = 5, 0

for epoch in range(start_epoch, EPOCHS+1):
    t0 = time.time()
    train_loss, train_acc, train_f1 = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_f1 = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | "
          f"train_loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
          f"val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")

    if val_f1 > best_val_f1 + 1e-4:
        best_val_f1 = val_f1
        no_improve = 0
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
        }, ckpt_path)
        torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
        print(f"[INFO] New best val F1: {val_f1:.4f}. Saved checkpoint.")
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
report = classification_report(all_y, all_p, target_names=SUBTYPES, digits=4)

print("\n=== TEST METRICS (8-class) ===")
print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
print("Confusion matrix:", cm)
print(report)

(Path(OUT_DIR) / "metrics.json").write_text(json.dumps(
    {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "report": report}, indent=2))
