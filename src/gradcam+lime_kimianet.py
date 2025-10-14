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

# --- LIME imports ---
from lime import lime_image
from skimage.segmentation import mark_boundaries

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

os.makedirs("runs/gradcam_kimianetmodel_outputs", exist_ok=True)

# --- PREDICTION HELPER FOR LIME ---
def predict_fn(images):
    """Takes a list of numpy images (H,W,3) and outputs model probabilities"""
    model.eval()
    batch = []
    for img in images:
        img_pil = Image.fromarray(img.astype(np.uint8))
        tensor = eval_tfms(img_pil).unsqueeze(0)
        batch.append(tensor)
    batch_tensor = torch.cat(batch).to(DEVICE)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return probs

# --- LOOP OVER IMAGES ---
for i, img_path in enumerate(sample_images):
    image = Image.open(img_path).convert("RGB")
    input_tensor = eval_tfms(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    # --- Model prediction ---
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    # --- Grad-CAM ---
    targets = [ClassifierOutputTarget(pred_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # --- LIME explanation ---
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(image.resize((224, 224))), 
        classifier_fn=predict_fn, 
        top_labels=NUM_CLASSES, 
        hide_color=0, 
        num_samples=1000
    )
    lime_img, mask = explanation.get_image_and_mask(
        label=pred_label,
        positive_only=False,
        hide_rest=False,
        num_features=10,
        min_weight=0.0
    )
    lime_img = mark_boundaries(lime_img / 255.0, mask)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM | Pred: {pred_label} | Conf: {confidence:.2f}")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(lime_img)
    plt.title("LIME")
    plt.axis("off")
    
    save_path = f"runs/gradcam_kimianetmodel_outputs/sample_{i}_pred{pred_label}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- CLEANUP ---
del cam
torch.cuda.empty_cache()

print("Grad-CAM + LIME visualizations saved to 'runs/gradcam_kimianetmodel_outputs/'")
