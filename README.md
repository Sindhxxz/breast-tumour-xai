
# Breast Tumour Analysis with Explainable AI (XAI)

This project performs **breast tumour classification** using histopathological images and integrates **Explainable AI (Grad-CAM, LIME, SHAP)** to visualize and interpret model decisions.  
It leverages the **BreakHis dataset** and deep learning architectures (CNNs / Vision Transformers) to build interpretable diagnostic systems.

---

## Setup

### Create a Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# On Windows:
venv\Scripts\activate
````

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧬 Dataset — BreakHis

**Source:** [BreakHis Dataset on Kaggle](https://www.kaggle.com/datasets/tathagatbanerjee/breakhis-breast-cancer-histopathological)
**Images:** 7,909 histopathology images
**Magnifications:** 40×, 100×, 200×, 400×
**Classes:** Benign / Malignant (with subtypes)

| Tumor Type    | Subtypes                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------ |
| **Benign**    | Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA)                       |
| **Malignant** | Ductal Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), Papillary Carcinoma (PC) |

Each image filename encodes biopsy method, class, subtype, patient ID, and magnification.
Example:
`SOB_B_TA-14-4659-40-001.png` → Benign, Tubular Adenoma, 40× magnification.

---

## ⚙️ Environment Requirements

* Python ≥ 3.9
* Framework: **PyTorch** (preferred) or TensorFlow
* Key libraries:

  * `torch`, `torchvision`
  * `scikit-learn`, `albumentations`
  * `grad-cam`, `shap`, `lime`
  * `streamlit` (for deployment/visualization)

---

## Project Workflow

### Step 1: Dataset Splitting — `split_by_patient.py`

* Shuffle and split patients into **train/val/test = 70/10/20**.
* Copy images into structured folders under `processed/`.
* Generate `manifest.csv` containing metadata for all images.
* Final counts:

  * **Train:** 5,463 images
  * **Val:** 733 images
  * **Test:** 1,713 images

---

### Step 2: Image Pre-processing — `preprocess_images.py`

* Resize every image to **224×224 px**.
* Convert to **RGB (3 channels)**.
* Save into `data/preprocessed/` maintaining the same split hierarchy.

---

### 🧠 Step 3: Dataset & DataLoader (PyTorch)

**Transforms**

```python
# Training
Resize(224, 224)
RandomHorizontalFlip()
RandomVerticalFlip()
ToTensor()
Normalize(mean=ImageNet_mean, std=ImageNet_std)

# Validation/Test
Resize(224, 224)
ToTensor()
Normalize(mean=ImageNet_mean, std=ImageNet_std)
```

**Dataset Loading**

```python
train_ds = ImageFolder(DATA_ROOT / "train", transform=train_tfms)
val_ds = ImageFolder(DATA_ROOT / "val", transform=eval_tfms)
test_ds = ImageFolder(DATA_ROOT / "test", transform=eval_tfms)
```

**Class Distribution**

| Split | Total | Benign | Malignant |
| ----- | ----- | ------ | --------- |
| Train | 5463  | 1622   | 3841      |
| Val   | 733   | 190    | 543       |
| Test  | 1713  | 668    | 1045      |

Observed class imbalance → handled during training via sampling or weighted loss.

**DataLoaders**

```python
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
```

Verified batch shape `[32, 3, 224, 224]` and correct label mapping.

---

## Folder Structure

```
breast-tumour-analysis-xai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── preprocessed/
│
├── runs/
│   
│
├── src/
│   ├── split_by_patient.py
│   ├── preprocess_images.py
│   ├── train.py
│
├── weights/
│   └── model.pt  # (ignored in .gitignore)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Explainable AI (XAI)

### Techniques Used:

* **Grad-CAM** → Visualizes important regions influencing model predictions.
* **LIME** → Generates local explanations for individual predictions.

Helps in validating model reliability and improving transparency in clinical contexts.

---

## 📚 Citation

If you use this repository or the BreakHis dataset, please cite:

> Spanhol, F.A., Oliveira, L.S., Petitjean, C., & Heutte, L.
> "A Dataset for Breast Cancer Histopathological Image Classification."
> *IEEE Transactions on Biomedical Engineering*, 2016.

---
