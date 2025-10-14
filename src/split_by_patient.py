import random
import shutil
import csv
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------
# CONFIG
# ---------------------------------------
SEED = 42
SRC_ROOT = Path("data/raw/BreaKHis_v1/histology_slides/breast")  # <- your confirmed path
DST_ROOT = Path("data/processed")
SPLITS = {"train": 0.70, "val": 0.10, "test": 0.20}

# Expected magnification directories (dataset standard)
MAG_DIRS = {"40X", "100X", "200X", "400X"}

# If True, keep per-patient subfolders in destination (False = flat within subtype)
KEEP_PATIENT_SUBFOLDERS = False

# ---------------------------------------
# UTILITIES
# ---------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def discover_layout(category_dir: Path) -> Tuple[bool, List[str]]:
    """
    Detect whether subtypes are under an 'SOB' folder or directly
    beneath the category. Returns (has_sob_layer, subtype_names).
    """
    sob_dir = category_dir / "SOB"
    if sob_dir.exists() and any(d.is_dir() for d in sob_dir.iterdir()):
        subtypes = sorted([d.name for d in sob_dir.iterdir() if d.is_dir()])
        return True, subtypes

    # Fallback: subtypes directly under the category
    subtypes = sorted([d.name for d in category_dir.iterdir() if d.is_dir()])
    return False, subtypes

def list_patient_dirs(category_dir: Path, subtype: str, has_sob: bool) -> List[Path]:
    """
    Return a list of patient directory Paths for the given subtype.
    Supports either .../<category>/SOB/<subtype>/<patient> or
    .../<category>/<subtype>/<patient>.
    """
    base = (category_dir / "SOB" / subtype) if has_sob else (category_dir / subtype)
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]

def split_patients(patient_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Shuffle and split into train/val/test by patient (70/10/20)."""
    random.shuffle(patient_dirs)
    n = len(patient_dirs)
    n_train = int(round(SPLITS["train"] * n))
    n_val   = int(round(SPLITS["val"]   * n))
    n_test  = n - n_train - n_val
    return {
        "train": patient_dirs[:n_train],
        "val":   patient_dirs[n_train:n_train+n_val],
        "test":  patient_dirs[n_train+n_val:],
    }

def iter_patient_pngs(patient_dir: Path) -> List[Path]:
    """
    Return all PNGs under known magnification subfolders; if none found,
    fall back to recursive search (covers variant layouts).
    """
    images = []
    for mag in MAG_DIRS:
        mag_dir = patient_dir / mag
        if mag_dir.exists():
            images.extend(mag_dir.glob("*.png"))
    if not images:
        images = list(patient_dir.rglob("*.png"))
    return images

def copy_images_for_patient(
    patient_dir: Path,
    dst_root_for_subtype: Path,
    manifest_writer: csv.DictWriter,
    split: str,
    category: str,
    subtype: str,
) -> int:
    """
    Copy all images for a given patient into the destination subtype folder.
    If KEEP_PATIENT_SUBFOLDERS is True, create subtype/<patient_id> in dst.
    Returns number of images copied.
    """
    imgs = iter_patient_pngs(patient_dir)
    if not imgs:
        return 0

    dst_dir = dst_root_for_subtype / (patient_dir.name if KEEP_PATIENT_SUBFOLDERS else "")
    ensure_dir(dst_dir)

    copied = 0
    for src in imgs:
        # Default destination filename
        dst = dst_dir / src.name
        # If a collision occurs (rare across different patients), disambiguate
        if dst.exists():
            dst = dst_dir / f"{patient_dir.name}__{src.name}"

        shutil.copy2(src, dst)
        copied += 1

        # Magnification best-effort (parent dir name if it matches)
        parent_name = src.parent.name
        magnification = parent_name if parent_name in MAG_DIRS else ""

        manifest_writer.writerow({
            "split": split,
            "category": category,
            "subtype": subtype,
            "patient_id": patient_dir.name,
            "magnification": magnification,
            "src_path": str(src),
            "dst_path": str(dst),
            "label_binary": 0 if category.lower() == "benign" else 1
        })

    return copied

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    random.seed(SEED)

    # Validate source root
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT.resolve()}")

    categories = []
    for cat in ("benign", "malignant"):
        cdir = SRC_ROOT / cat
        if cdir.exists() and cdir.is_dir():
            categories.append(cat)
        else:
            print(f"[WARN] Missing category directory: {cdir}")

    if not categories:
        raise RuntimeError("No categories found (expected 'benign' and/or 'malignant').")

    # Prepare destination skeleton (top-level only)
    for split in SPLITS.keys():
        for category in categories:
            ensure_dir(DST_ROOT / split / category)

    # Open manifest CSV
    ensure_dir(DST_ROOT)
    manifest_path = DST_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split", "category", "subtype", "patient_id",
            "magnification", "src_path", "dst_path", "label_binary"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_imgs = {"train": 0, "val": 0, "test": 0}
        total_patients = {"train": 0, "val": 0, "test": 0}

        for category in categories:
            category_dir = SRC_ROOT / category
            has_sob, subtypes = discover_layout(category_dir)

            if not subtypes:
                print(f"[WARN] No subtypes discovered under: {category_dir}")
                continue

            print(f"[INFO] Category='{category}' | has_SOB_layer={has_sob} | subtypes={subtypes}")

            # Ensure subtype destination folders exist
            for split in SPLITS.keys():
                for st in subtypes:
                    ensure_dir(DST_ROOT / split / category / st)

            for st in subtypes:
                patient_dirs = list_patient_dirs(category_dir, st, has_sob)
                if not patient_dirs:
                    print(f"[WARN] No patient folders found: {category}/{st}")
                    continue

                splits = split_patients(patient_dirs)

                for split, pts in splits.items():
                    dst_subtype_root = DST_ROOT / split / category / st
                    for pdir in pts:
                        copied = copy_images_for_patient(
                            pdir, dst_subtype_root, writer, split, category, st
                        )
                        if copied > 0:
                            total_patients[split] += 1
                            total_imgs[split] += copied

        print("\n[SUMMARY]")
        for split in ("train", "val", "test"):
            print(f"  {split}: patients={total_patients[split]:5d}  images={total_imgs[split]:7d}")
        print(f"\nManifest written to: {manifest_path.resolve()}")

if __name__ == "__main__":
    main()
