# train_kimianet_binary_v2.py
"""
Upgraded training script based on user's baseline.
Includes:
 - tuned augmentations (RandomResizedCrop, rotations, blur)
 - progressive unfreezing (unfreeze last DenseNet block + classifier; option to unfreeze more later)
 - FocalLoss (for class imbalance)
 - OneCycleLR scheduler
 - mixed-precision training (torch.cuda.amp)
 - optional WeightedRandomSampler (commented) for strong class balancing
 - validation threshold tuning for binary F1
 - checkpointing and best-model saving

Notes:
 - Tune MAX_LR and IMG_SIZE to your GPU resources.
 - If you want to unfreeze more layers sooner, change UNFREEZE_AFTER_EPOCH.
"""

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
DATA_ROOT = Path("data/preprocessed")
OUT_DIR   = Path("runs/kimiaNet_binary_v2"); OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = Path("weights/KimiaNetPyTorchWeights.pth")
NUM_CLASSES = 2  # Binary

FREEZE_BACKBONE = True           # start with backbone frozen
UNFREEZE_AFTER_EPOCH = 5        # epoch after which we'll unfreeze last block (set None to disable)
EPOCHS = 50
BATCH_SIZE = 16
BASE_LR = 1e-5                  # base lr for fine-tuning head
MAX_LR = 1e-3                   # OneCycleLR max_lr (tune this)
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = (256, 256)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["benign", "malignant"]

# --------------------
# Dataset
# --------------------
def collect_split(split_root: Path):
    img_paths, labels = [], []
    for label, cat in enumerate(CLASS_NAMES):  # 0 = benign, 1 = malignant
        cat_dir = split_root / cat
        if not cat_dir.exists():
            continue
        for root, _, files in os.walk(cat_dir):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                    img_paths.append(Path(root) / fn)
                    labels.append(label)
    return img_paths, labels

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, split_root: Path, transform=None):
        self.samples, self.targets = collect_split(split_root)
        self.transform = transform

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
# Transforms (tuned)
# --------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.GaussianBlur(kernel_size=(3, 5)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Build datasets & loaders
train_ds = BinaryDataset(DATA_ROOT / "train", transform=train_tfms)
val_ds   = BinaryDataset(DATA_ROOT / "val",   transform=eval_tfms)
test_ds  = BinaryDataset(DATA_ROOT / "test",  transform=eval_tfms)

# Compute class weights for loss
train_targets = np.array(train_ds.targets)
class_counts  = np.bincount(train_targets, minlength=NUM_CLASSES)
class_weights = class_counts.sum() / (len(class_counts) * np.clip(class_counts, 1, None))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# Optional: WeightedRandomSampler to balance classes at batch-level
use_weighted_sampler = False
if use_weighted_sampler:
    class_sample_count = np.bincount(train_ds.targets)
    weights = 1.0 / class_sample_count
    sample_weights = [weights[t] for t in train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

val_loader  = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

# --------------------
# Model builder (kimianet weights loaded into DenseNet121)
# --------------------

def build_kimianet(weights_path: Path, num_classes: int):
    model = models.densenet121(weights=None)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    
    return model.to(DEVICE)

if __name__ == "__main__":

    model = build_kimianet(WEIGHTS_PATH, NUM_CLASSES)

    # --------------------
    # Freezing / progressive unfreeze utilities
    # --------------------

    def freeze_backbone(m):
        for name, param in m.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        for param in m.classifier.parameters():
            param.requires_grad = True

    # Unfreeze last DenseNet block(s) - for densenet121 the "denseblock4" is the final block

    def unfreeze_last_block(m):
        for name, param in m.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True

    if FREEZE_BACKBONE:
        freeze_backbone(model)

    # --------------------
    # Focal loss (binary/categorical)
    # --------------------
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=None, reduction='mean'):  # alpha can be tensor of class weights
            super().__init__()
            self.gamma = gamma
            if alpha is not None:
                self.alpha = torch.tensor(alpha, dtype=torch.float32).to(DEVICE)
            else:
                self.alpha = None
            self.reduction = reduction

        def forward(self, logits, targets):
            # logits: (N, C), targets: (N,) int
            logpt = -nn.functional.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(logpt)
            loss = -((1 - pt) ** self.gamma) * logpt
            if self.alpha is not None:
                at = self.alpha[targets]
                loss = at * loss
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

    criterion = FocalLoss(gamma=2.0, alpha=class_weights)

    # --------------------
    # Optimizer + OneCycleLR
    # --------------------
    # Only parameters with requires_grad=True will be optimized
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
    # OneCycleLR requires steps_per_epoch
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR,
                                                    steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                                                    pct_start=0.3, anneal_strategy='cos', cycle_momentum=False)

    # --------------------
    # Resume Training (loads checkpoint if present)
    # --------------------
    start_epoch = 1
    ckpt_path = OUT_DIR / "checkpoint.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # Note: OneCycleLR state is not restored here; safe to restart but lr schedule will start fresh
        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resuming from epoch {start_epoch}")

    # --------------------
    # Mixed precision setup
    # --------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    # --------------------
    # Training / Eval loops
    # --------------------

    def compute_threshold_and_metrics(y_true, y_probs):
        # Find threshold in [0.3,0.7] that maximizes macro F1 on validation set
        best_t, best_f1 = 0.5, 0.0
        thresholds = np.linspace(0.3, 0.7, 41)
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t, best_f1


    def run_epoch(loader, train: bool):
        model.train(train)
        losses, all_y, all_p, all_prob_pos = [], [], [], []

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            if train:
                optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # Step scheduler per batch for OneCycleLR
                scheduler.step()

            losses.append(loss.item())
            probs = nn.functional.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # positive-class prob
            preds = (probs >= 0.5).astype(int)

            all_y.extend(y.detach().cpu().numpy())
            all_p.extend(preds)
            all_prob_pos.extend(probs)

        acc = accuracy_score(all_y, all_p)
        f1  = f1_score(all_y, all_p, average="macro")
        return float(np.mean(losses)), acc, f1, np.array(all_y), np.array(all_prob_pos)

    best_val_f1 = 0.0
    patience, no_improve = 10, 0

    for epoch in range(start_epoch, EPOCHS+1):
        t0 = time.time()

        # If progressive unfreeze is enabled, unfreeze after configured epoch
        if FREEZE_BACKBONE and UNFREEZE_AFTER_EPOCH is not None and epoch == UNFREEZE_AFTER_EPOCH:
            print(f"[INFO] Unfreezing last block at epoch {epoch}")
            unfreeze_last_block(model)
            # Recreate optimizer to pick up new parameters
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
            # Re-create scheduler for remaining epochs (OneCycleLR needs total epochs)
            remaining_epochs = max(1, EPOCHS - epoch + 1)
            steps_per_epoch = max(1, len(train_loader))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR,
                                                            steps_per_epoch=steps_per_epoch, epochs=remaining_epochs,
                                                            pct_start=0.3, anneal_strategy='cos', cycle_momentum=False)

        train_loss, train_acc, train_f1, train_y, train_prob = run_epoch(train_loader, train=True)

        # Validation: run in eval mode but we will collect probs to tune threshold
        model.train(False)
        val_losses, val_y_all, val_prob_all = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_losses.append(loss.item())
                probs = nn.functional.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                val_prob_all.extend(probs)
                val_y_all.extend(y.detach().cpu().numpy())

        val_y_all = np.array(val_y_all)
        val_prob_all = np.array(val_prob_all)
        # tune threshold on validation
        best_t, tuned_val_f1 = compute_threshold_and_metrics(val_y_all, val_prob_all)
        val_preds_tuned = (val_prob_all >= best_t).astype(int)
        val_acc = accuracy_score(val_y_all, val_preds_tuned)

        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.3f} tuned_f1 {tuned_val_f1:.3f} thresh {best_t:.2f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")

        # Save checkpoint & best model
        if tuned_val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = tuned_val_f1
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "threshold": float(best_t)
            }, ckpt_path)
            torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
            print(f"[INFO] New best val F1: {tuned_val_f1:.4f}. Saved checkpoint.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[INFO] Early stopping.")
                break

    # --------------------
    # Final Test Evaluation (use tuned threshold from checkpoint if saved)
    # --------------------

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if "threshold" in ckpt:
            best_threshold = ckpt["threshold"]
        else:
            best_threshold = 0.5
        model.load_state_dict(torch.load(OUT_DIR / "best_model.pth", map_location=DEVICE))
    else:
        best_threshold = 0.5

    model.eval()
    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= best_threshold).astype(int)
            all_prob.extend(probs)
            all_p.extend(preds)
            all_y.extend(y.numpy())

    acc = accuracy_score(all_y, all_p)
    f1  = f1_score(all_y, all_p, average="macro")
    cm  = confusion_matrix(all_y, all_p).tolist()
    report = classification_report(all_y, all_p, target_names=CLASS_NAMES, digits=4)

    print("\n=== TEST METRICS (Binary) ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | thresh: {best_threshold}")
    print("Confusion matrix:", cm)
    print(report)

    (Path(OUT_DIR) / "metrics.json").write_text(json.dumps(
        {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "report": report, "threshold": best_threshold}, indent=2))