import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- CONFIG ---
DEVICE = torch.device("cpu")  # running on CPU
NUM_CLASSES = 2
WEIGHTS_PATH = "runs/kimiaNet_binary_v2/best_model.pth"

# --- same normalization as during training ---
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- pick an example image ---
img_path = "data/preprocessed/test/malignant/ductal_carcinoma/SOB_M_DC-14-2773-40-001.png"
image = Image.open(img_path).convert('RGB')
input_tensor = eval_tfms(image).unsqueeze(0).to(DEVICE)

# --- load trained model without triggering optimizer/training code ---
from kimianet_binary_v2 import build_kimianet
model = build_kimianet(weights_path=WEIGHTS_PATH, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- choose target layer (last convolutional block) ---
# for EfficientNet/DenseNet-based KimiaNet, usually last conv block is fine
target_layers = [model.features[-1]]

# wrap input
input_tensor = eval_tfms(image).unsqueeze(0).to(DEVICE)
input_tensor.requires_grad = True

# Grad-CAM
cam = GradCAM(model=model, target_layers=[model.features[-1]])
targets = [ClassifierOutputTarget(1)]  # malignant

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0]

# overlay heatmap
rgb_img = np.array(image.resize((224, 224))) / 255.0
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(rgb_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cam_image)
plt.title("Grad-CAM")
plt.axis("off")
plt.show()
