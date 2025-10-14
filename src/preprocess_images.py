from pathlib import Path
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path("data/processed")      # input folder
DST_ROOT = Path("data/preprocessed")   # output folder
TARGET_SIZE = (224, 224)

def resize_and_save(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize(TARGET_SIZE, Image.BILINEAR)
        im.save(dst, format="PNG")

def main():
    images = list(SRC_ROOT.rglob("*.png"))  # find all .png images
    print(f"Found {len(images)} images")

    for src in tqdm(images, desc="Resizing"):
        rel_path = src.relative_to(SRC_ROOT)
        dst = DST_ROOT / rel_path
        resize_and_save(src, dst)

    print("All images resized and saved to", DST_ROOT)

if __name__ == "__main__":
    main()
