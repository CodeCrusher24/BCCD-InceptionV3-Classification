# Blood Cell Classification Using InceptionV3

> B.Tech CSE | Deep Learning Assignment | SoCSE, SMVDU | Semester 6 (May 2026)

---

## Authors

| Name | Entry No. |
|---|---|
| Parag Kumar | 23BCS060 |
| Rohit Yadav | 23BCS077 |
| Rishav Kumar Gupta | 23BCS076 |

---

## Overview

This project performs **four-class blood cell image classification** using **InceptionV3 full fine-tuning** in PyTorch on the **BCCD dataset**.

| Feature | Details |
|---|---|
| Model | InceptionV3 (ImageNet pretrained, fully fine-tuned) |
| Dataset | BCCD Blood Cell Dataset |
| Classes | Eosinophil, Lymphocyte, Monocyte, Neutrophil |
| Framework | PyTorch |
| Platform | Google Colab (T4 GPU) |
| Input Size | 299 × 299 |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **100.00%** |
| Macro F1-Score | **1.00** |
| Macro AUC-ROC | **1.0000** |
| Epochs Trained | 21 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Eosinophil | 1.00 | 1.00 | 1.00 | 1.0000 |
| Lymphocyte | 1.00 | 1.00 | 1.00 | 1.0000 |
| Monocyte | 1.00 | 1.00 | 1.00 | 1.0000 |
| Neutrophil | 1.00 | 1.00 | 1.00 | 1.0000 |

---

## Frozen Backbone vs Full Fine-Tuning

| Metric | Frozen Backbone | Full Fine-Tuning |
|---|---|---|
| Test Accuracy | 61.07% | **100.00%** |
| Macro F1 | 0.61 | **1.00** |
| Macro AUC-ROC | 0.8434 | **1.0000** |

---

## Output Visualizations

### Training & Validation Curves
![Training Plot](outputs/train_val_plot.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

### ROC Curves
![ROC Curves](outputs/roc_curves.png)

---

## Repository Structure

```text
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── train_val_plot.png
├── paper/
├── .gitattributes
├── InceptionV3BCCD.py
└── README.md
```

---

## Dataset

Download the BCCD dataset from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/blood-cells

---

## Dataset Structure

Organize the dataset in the following format before training:

```text
data/
├── EOSINOPHIL/
├── LYMPHOCYTE/
├── MONOCYTE/
└── NEUTROPHIL/
```

---

## Installation

Install the required dependencies:

```bash
pip install torch torchvision tqdm scikit-learn seaborn matplotlib
```

---

## Run the Training Script

```bash
python InceptionV3BCCD.py
```

Training runtime is approximately **2.5 hours** on a Google Colab T4 GPU.

---

## Model Configuration

| Component | Details |
|---|---|
| Backbone | InceptionV3 |
| Trainable Parameters | 24,354,536 |
| Final Layer | `Linear(2048, 4)` |
| Auxiliary Layer | `Linear(768, 4)` |
| Loss Function | Cross-Entropy with Label Smoothing |
| Label Smoothing | ε = 0.1 |
| Auxiliary Loss Weight | 0.4 × aux loss |
| Optimizer | Adam |
| Learning Rate | 1e-5 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Minimum LR | 1e-7 |
| Gradient Clipping | Max norm = 1.0 |
| Early Stopping | Patience = 7 |

---

## Features

- Full InceptionV3 fine-tuning
- Transfer learning using ImageNet pretrained weights
- Label smoothing regularization
- Cosine annealing learning-rate scheduling
- Auxiliary classifier support
- Early stopping mechanism
- ROC-AUC evaluation
- Confusion matrix visualization

---

## Conclusion

The fully fine-tuned InceptionV3 model achieved perfect classification performance on the BCCD dataset, significantly outperforming the frozen-backbone baseline.

---

*May 2026 | B.Tech CSE 6th Semester | SoCSE, SMVDU*
