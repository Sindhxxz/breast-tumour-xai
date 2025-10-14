import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from kimianet_binary_v2 import build_kimianet
import random

# --- CONFIG ---
DEVICE = torch.device("cpu")  # or "cuda" if GPU available
NUM_CLASSES = 2
WEIGHTS_PATH = "runs/kimiaNet_binary_v2/best_model.pth"

# --- TRANSFORMS ---
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD MODEL ---
model = build_kimianet(weights_path=WEIGHTS_PATH, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Grad-CAM SETUP ---
# ⚠️ No 'use_cuda' in latest version
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# --- TEST IMAGE SELECTION ---
TEST_DIR = "data/preprocessed/test/"
CLASSES = ["benign", "malignant"]

sample_images = []
for cls in CLASSES:
    class_path = os.path.join(TEST_DIR, cls)
    for subdir in os.listdir(class_path):
        subdir_path = os.path.join(class_path, subdir)
        if os.path.isdir(subdir_path):
            imgs = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".png")]
            sample_images.extend(random.sample(imgs, min(5, len(imgs))))

print(f"Selected {len(sample_images)} test images.")

os.makedirs("runs/gradcam_kimianet_outputs", exist_ok=True)

# --- LOOP OVER IMAGES ---
for i, img_path in enumerate(sample_images):
    image = Image.open(img_path).convert("RGB")
    input_tensor = eval_tfms(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    # Grad-CAM calculation
    targets = [ClassifierOutputTarget(pred_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    rgb_img = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM | Pred: {pred_label} | Conf: {confidence:.2f}")
    plt.axis("off")

    save_path = f"runs/gradcam_kimianet_outputs/sample_{i}_pred{pred_label}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Proper cleanup (fixes AttributeError warning)
del cam
torch.cuda.empty_cache()

print("Grad-CAM visualizations saved to 'runs/gradcam_kimianet_outputs/'")
