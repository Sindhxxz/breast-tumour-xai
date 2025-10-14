import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- CONFIG ---
DEVICE = torch.device("cpu")
NUM_CLASSES = 2
WEIGHTS_PATH = "runs/kimiaNet_binary_v2/best_model.pth"

# --- preprocessing ---
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- load model ---
from kimianet_binary_v2 import build_kimianet
model = build_kimianet(weights_path=WEIGHTS_PATH, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- load test image ---
img_path = "data/preprocessed/test/malignant/ductal_carcinoma/SOB_M_DC-14-2773-40-001.png"
image = Image.open(img_path).convert('RGB')
input_tensor = eval_tfms(image).unsqueeze(0).to(DEVICE)

# --- Grad-CAM ---
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(1)]  # malignant

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
rgb_img = np.array(image.resize((224, 224))) / 255.0
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# --- LIME ---
def batch_predict(images):
    model.eval()
    batch = torch.stack([eval_tfms(Image.fromarray(img)) for img in images], dim=0).to(DEVICE)
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    np.array(image.resize((224, 224))),       # input image
    batch_predict,                            # prediction function
    top_labels=2,
    hide_color=0,
    num_samples=1000                          # number of perturbations
)

# get the explanation for predicted label
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)
lime_img = mark_boundaries(temp / 255.0, mask)

# --- visualize both side by side ---
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(rgb_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cam_image)
plt.title("Grad-CAM")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(lime_img)
plt.title("LIME")
plt.axis("off")

plt.tight_layout()
plt.show()
