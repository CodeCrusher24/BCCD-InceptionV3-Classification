# Blood Cell Classification Using InceptionV3

> B.Tech CSE | Deep Learning Assignment | SoCSE, SMVDU | Semester 6 (2025–26)

---

## Authors
| Name | Entry No. |
|---|---|
| Parag Kumar | 23BCS060 |
| Ojas Kumar | 23BCS059 |
| Akaisha Sundhan | 23BCS009 |

---

## Overview
Four-class blood cell image classification using **InceptionV3 transfer learning** in PyTorch on the BCCD dataset.

| | |
|---|---|
| **Model** | InceptionV3 (ImageNet pretrained) |
| **Dataset** | BCCD — 9,957 microscopic blood cell images |
| **Classes** | Eosinophil · Lymphocyte · Monocyte · Neutrophil |
| **Framework** | PyTorch |
| **Platform** | Google Colab (T4 GPU) |
| **Input Size** | 299 × 299 |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **61.07%** |
| Macro F1-Score | **0.61** |
| Macro AUC-ROC | **0.8434** |

| Class | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Eosinophil | 0.61 | 0.40 | 0.48 | 0.8125 |
| Lymphocyte | 0.59 | 0.75 | 0.66 | 0.8730 |
| Monocyte | 0.76 | 0.65 | 0.70 | 0.8950 |
| Neutrophil | 0.53 | 0.66 | 0.58 | 0.7932 |

---

## Outputs

![Training Plot](outputs/train_val_plot.png)

![Confusion Matrix](outputs/confusion_matrix.png)

![ROC Curves](outputs/roc_curves.png)

---

## Repository Structure

```
├── BCCD_InceptionV3.py         # Full training script (run in Google Colab)
├── model.pth                   # Trained model weights
├── outputs/
│   ├── train_val_plot.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
├── paper/
│   ├── main.tex                # IEEE LaTeX source
│   ├── DeepLearningAssignment.pdf
│   ├── train_val_plot.png
│   ├── confusion_matrix.png
│   └── roc_curves.png
└── README.md
```

---

## How to Run

### 1. Get the dataset
Download from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/blood-cells

### 2. Set up Colab

**Cell 1 — Install:**
```python
!pip install -q torch torchvision tqdm scikit-learn seaborn matplotlib
```

**Cell 2 — Extract dataset:**
```python
import zipfile, shutil, os
with zipfile.ZipFile("/content/archive.zip", "r") as z:
    z.extractall("/content/raw")
shutil.copytree(
    "/content/raw/dataset2-master/dataset2-master/images/TRAIN",
    "/content/data"
)
print(os.listdir("/content/data"))
# Expected: ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
```

**Cell 3 — Run** `BCCD_InceptionV3.py`

Runtime: ~18 minutes on T4 GPU.

---

## Model Details

- Backbone frozen (all conv layers)
- Trainable: `model.fc` → `Linear(2048, 4)` and `model.AuxLogits.fc` → `Linear(768, 4)`
- Loss: Cross-entropy + 0.4 × auxiliary loss
- Optimizer: Adam, lr = 1e-4
- Scheduler: StepLR (γ = 0.5, every 3 epochs)
- Early stopping: patience = 3 on val accuracy
