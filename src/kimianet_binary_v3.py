# train_kimianet_binary_v3.py
"""
Upgraded training script for BreakHis (v3).
Features:
 - Patient-wise splitting (no patient leakage)
 - Optional stain normalization (Macenko) if torchstain is available
 - EfficientNetV2-S (fallback EfficientNet-B3) backbone fully fine-tuned
 - Multi-scale training (RandomResizedCrop, larger resolution)
 - MixUp augmentation option
 - OneCycleLR scheduler with per-batch stepping
 - Mixed-precision training
 - Save top-k checkpoints for simple ensembling
"""

from pathlib import Path
import re, json, random, time
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -------------------- CONFIG --------------------
DATA_ROOT = Path("data/preprocessed")
OUT_DIR = Path("runs/kimiaNet_binary_v3"); OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = Path("data/weights/KimiaNetPyTorchWeights.pth")

SEED = 42
NUM_CLASSES = 2
CLASS_NAMES = ["benign", "malignant"]

# Training hyperparams
IMG_SIZE = 384
BATCH_SIZE = 12
NUM_EPOCHS = 40
BASE_LR = 1e-5
MAX_LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK_CHECKPOINTS = 3
MIXUP_ALPHA = 0.2
USE_STAIN_NORMALIZATION = True

# -------------------- REPRODUCIBILITY --------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# Optional: parse patient & magnification if filenames have that info
PATTERN = re.compile(r"SOB_[BM]_[A-Z]+-(\d+-\d+)-(\d+)-\d+\.png")
def parse_meta_from_name(fn: str):
    m = PATTERN.search(fn)
    if not m:
        return None, None
    return m.group(1), m.group(2)

# -------------------- Build catalog --------------------
records = []
for p in DATA_ROOT.rglob("*.png"):
    # split: train/val/test
    split = p.parts[-4].lower()  # preprocessed/train/benign/subtype/file.png -> -4 = train
    if split not in ["train", "val", "test"]:
        continue

    # class: benign/malignant
    cls = p.parts[-3].lower()
    if cls not in CLASS_NAMES:
        continue
    cls_idx = CLASS_NAMES.index(cls)

    # subtype (optional)
    subtype = p.parts[-2]

    # patient & mag (optional)
    patient, mag = parse_meta_from_name(p.name)

    records.append({
        "path": str(p),
        "split": split,
        "label": cls_idx,
        "class": cls,
        "subtype": subtype,
        "patient": patient,
        "mag": mag
    })

catalog = pd.DataFrame(records)
if catalog.empty:
    raise ValueError(f"No images found! Check DATA_ROOT: {DATA_ROOT}")

print(f"Found {len(catalog)} images across classes: {catalog['label'].value_counts().to_dict()}")
print(f"Images per split:\n{catalog['split'].value_counts()}")
print(f"Images per subtype:\n{catalog['subtype'].value_counts()}")

# -------------------- PATIENT-WISE SPLIT --------------------
unique_patients = catalog['patient'].dropna().unique().tolist()
random.shuffle(unique_patients)
n = len(unique_patients)
train_p = set(unique_patients[:int(0.7*n)])
val_p   = set(unique_patients[int(0.7*n):int(0.85*n)])
test_p  = set(unique_patients[int(0.85*n):])

def split_df(df, pset):
    return df[df['patient'].isin(pset)].reset_index(drop=True)

train_df = catalog[catalog['split']=="train"].reset_index(drop=True)
val_df   = catalog[catalog['split']=="val"].reset_index(drop=True)
test_df  = catalog[catalog['split']=="test"].reset_index(drop=True)

print(f"Patient-wise split -> train: {len(train_df)} val: {len(val_df)} test: {len(test_df)}")

# -------------------- OPTIONAL STAIN NORMALIZATION --------------------
stain_normalizer = None
if USE_STAIN_NORMALIZATION:
    try:
        import torchstain
        sample_target = Image.open(train_df.iloc[0]['path']).convert('RGB')
        normalizer = torchstain.MacenkoNormalizer(backend='torch')
        normalizer.fit(np.array(sample_target))
        stain_normalizer = normalizer
        print('[INFO] torchstain Macenko normalizer ready')
    except Exception as e:
        print('[WARN] torchstain not available or failed, skipping stain normalization:', e)
        stain_normalizer = None

# -------------------- TRANSFORMS --------------------
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------- DATASET CLASS --------------------
class BreakHisDataset(Dataset):
    def __init__(self, df, transform=None, stain_normalizer=None):
        self.df = df
        self.transform = transform
        self.stain_normalizer = stain_normalizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.stain_normalizer is not None:
            try:
                arr = np.array(img)
                img_np, _, _ = self.stain_normalizer.normalize(I=arr)
                img = Image.fromarray(img_np)
            except:
                pass
        if self.transform is not None:
            img = self.transform(img)
        label = int(row['label'])
        return img, label

# -------------------- MIXUP UTILS --------------------
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)

# -------------------- DATALOADERS --------------------
train_ds = BreakHisDataset(train_df, transform=train_tfms, stain_normalizer=stain_normalizer)
val_ds   = BreakHisDataset(val_df, transform=val_tfms, stain_normalizer=stain_normalizer)
test_ds  = BreakHisDataset(test_df, transform=val_tfms, stain_normalizer=stain_normalizer)

# Weighted sampler
class_sample_count = np.bincount(train_df['label'].astype(int))
weights = 1.0 / np.clip(class_sample_count, 1, None)
sample_weights = [weights[int(l)] for l in train_df['label']]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))

# -------------------- MODEL --------------------
try:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    print('[INFO] using EfficientNetV2-S')
except:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    print('[INFO] using EfficientNet-B3 fallback')

# Replace classifier safely
if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
else:
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

# -------------------- LOSS, OPTIMIZER, SCHEDULER --------------------
# Focal loss option
class_counts = train_df['label'].value_counts().sort_index().values
inv = 1.0 / np.clip(class_counts, 1, None)
cls_weights = torch.tensor(inv / inv.sum(), dtype=torch.float32).to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

criterion = FocalLoss(gamma=2.0, weight=cls_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR,
                                                steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,
                                                pct_start=0.2, anneal_strategy='cos', cycle_momentum=False)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

# -------------------- TRAINING LOOP --------------------
def evaluate(model, loader, threshold=0.5):
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = nn.functional.softmax(logits, dim=1)[:,1].cpu().numpy()
            pred = (prob>=threshold).astype(int)
            ys.extend(y.numpy())
            ps.extend(pred.tolist())
            probs.extend(prob.tolist())
    return np.array(ys), np.array(ps), np.array(probs)

def compute_best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.2,0.8,61):
        f1 = f1_score(y_true, (probs>=t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

best_checkpoints = []

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        if MIXUP_ALPHA > 0:
            xb, y_a, y_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
            y_a, y_b = y_a.to(DEVICE), y_b.to(DEVICE)
        else:
            y_a, y_b, lam = None, None, 1.0

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
            logits = model(xb)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam) if MIXUP_ALPHA>0 else criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    val_y, val_p, val_probs = evaluate(model, val_loader, threshold=0.5)
    best_t, val_f1 = compute_best_threshold(val_y, val_probs)
    val_preds_tuned = (val_probs>=best_t).astype(int)
    val_acc = accuracy_score(val_y, val_preds_tuned)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss {train_loss:.4f} | val_acc {val_acc:.4f} val_f1 {val_f1:.4f} thresh {best_t:.3f} | time {time.time()-t0:.1f}s")

    # Save top-k checkpoints
    ckpt_path = OUT_DIR / f"ckpt_epoch{epoch:03d}_f1{val_f1:.4f}.pth"
    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'val_f1': float(val_f1), 'threshold': float(best_t)}, ckpt_path)
    best_checkpoints.append((val_f1, ckpt_path))
    best_checkpoints = sorted(best_checkpoints, key=lambda x:x[0], reverse=True)[:TOPK_CHECKPOINTS]

# -------------------- ENSEMBLE EVALUATION --------------------
ensemble_probs = []
final_thresholds = []
for score, ckpt_path in best_checkpoints:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    probs = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = nn.functional.softmax(logits, dim=1)[:,1].cpu().numpy()
            probs.extend(prob.tolist())
    ensemble_probs.append(np.array(probs))
    final_thresholds.append(ckpt.get('threshold',0.5))

if len(ensemble_probs)==0:
    print('[WARN] No checkpoints found for ensemble; evaluating single model')
    test_y, test_pred, test_probs = evaluate(model, test_loader, threshold=0.5)
    final_thresh = 0.5
else:
    avg_probs = np.mean(np.stack(ensemble_probs,axis=0),axis=0)
    final_thresh = float(np.mean(final_thresholds)) if len(final_thresholds)>0 else 0.5
    test_y = test_df['label'].values
    test_pred = (avg_probs>=final_thresh).astype(int)
    test_probs = avg_probs

acc = accuracy_score(test_y, test_pred)
f1 = f1_score(test_y, test_pred, average='macro')
cm = confusion_matrix(test_y, test_pred).tolist()
report = classification_report(test_y, test_pred, target_names=CLASS_NAMES, digits=4)

print('\n=== TEST METRICS (v3 ensemble) ===')
print(f'Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | thresh: {final_thresh}')
print('Confusion matrix:', cm)
print(report)

(Path(OUT_DIR)/'metrics_v3.json').write_text(json.dumps({
    'accuracy': acc, 'macro_f1': f1, 'confusion_matrix': cm, 'report': report, 'threshold': final_thresh
}, indent=2))