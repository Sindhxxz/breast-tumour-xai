from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from collections import Counter

# ---- Paths ----
DATA_ROOT = Path("data/preprocessed") 

# ---- Hyperparams ----
BATCH_SIZE   = 32
NUM_WORKERS  = 4        
IMG_SIZE     = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---- Transforms ----
train_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---- Datasets ----
train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=eval_tfms)
test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=eval_tfms)

print("\n Dataset sizes:")
print(f"  Train: {len(train_ds)} images")
print(f"  Val  : {len(val_ds)} images")
print(f"  Test : {len(test_ds)} images")

print("\n Class mapping:", train_ds.class_to_idx)

def count_per_class(ds, name):
    counts = Counter([label for _, label in ds.samples])
    print(f"{name} per-class counts:")
    for cls, idx in ds.class_to_idx.items():
        print(f"  {cls:10s}: {counts[idx]}")

count_per_class(train_ds, "Train")
count_per_class(val_ds,   "Val")
count_per_class(test_ds,  "Test")

# ---- DataLoaders ----
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ---- Sanity check: look at one batch ----
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, labels = next(iter(train_loader))
    print("\n One batch:")
    print("  Images shape :", imgs.shape)   
    print("  Labels shape :", labels.shape)
    print("  First labels :", labels[:8].tolist())
    print("  Device       :", device)
