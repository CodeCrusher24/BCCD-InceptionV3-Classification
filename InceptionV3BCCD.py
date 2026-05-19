# Blood Cell Classification - InceptionV3
# BCCD Dataset - 4 classes
# Done in Google Colab with T4 GPU

# install stuff first if needed
# !pip install torch torchvision tqdm scikit-learn seaborn matplotlib

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import Inception_V3_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ------- settings -------
data_path   = "/content/data"
n_classes   = 4
img_size    = 299       # inception needs 299x299
epochs      = 30
lr          = 1e-5      # keeping it small since we're fine tuning everything
batch       = 16        # 32 was causing memory issues with full model
split_train = 0.70
split_val   = 0.15
# test gets the remaining 0.15
stop_after  = 7         # early stopping patience
out_model   = "model_v2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ------- transforms -------
# training gets augmentation, val/test just gets resized + normalized
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    # imagenet mean and std
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ------- load dataset -------
full_data   = datasets.ImageFolder(root=data_path, transform=train_tf)
labels      = full_data.classes
print("classes:", labels)
print("total images:", len(full_data))

total   = len(full_data)
n_train = int(total * split_train)
n_val   = int(total * split_val)
n_test  = total - n_train - n_val

# fixed seed so split is reproducible
gen = torch.Generator().manual_seed(42)
train_split, val_split, test_split = random_split(full_data, [n_train, n_val, n_test], generator=gen)

# wrapper so we can apply different transforms per split
class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, split, tf):
        self.split = split
        self.tf    = tf

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        from PIL import Image
        # get original file path from parent dataset
        idx        = self.split.indices[i]
        path, lbl  = self.split.dataset.samples[idx]
        img        = Image.open(path).convert("RGB")
        return self.tf(img), lbl

train_ds = SplitDataset(train_split, train_tf)
val_ds   = SplitDataset(val_split,   test_tf)
test_ds  = SplitDataset(test_split,  test_tf)

train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

print(f"train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")


# ------- model setup -------
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

# swap out the final layers for our 4 classes
model.fc           = nn.Linear(model.fc.in_features, n_classes)
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, n_classes)

# unfreeze everything - full fine tuning
for p in model.parameters():
    p.requires_grad = True

model = model.to(device)

total_p = sum(p.numel() for p in model.parameters())
print(f"total params: {total_p:,}")


# ------- loss and optimizer -------
# label smoothing helps prevent overconfident predictions
loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# cosine annealing - lr gradually decreases to near zero
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)


# ------- training -------
log = {"tloss": [], "tacc": [], "vloss": [], "vacc": []}

best_acc   = 0.0
best_wts   = copy.deepcopy(model.state_dict())
no_improve = 0

for ep in range(1, epochs + 1):
    cur_lr = scheduler.get_last_lr()[0]
    print(f"\nepoch {ep}/{epochs}  lr={cur_lr:.2e}")
    print("-" * 40)

    # -- train --
    model.train()
    t_loss    = 0.0
    t_correct = 0

    for imgs, lbls in tqdm(train_dl, desc="train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()

        out = model(imgs)

        # inception returns (main, aux) during training
        if isinstance(out, tuple):
            main, aux = out
            loss = loss_fn(main, lbls) + 0.4 * loss_fn(aux, lbls)
            pred = main.argmax(dim=1)
        else:
            loss = loss_fn(out, lbls)
            pred = out.argmax(dim=1)

        loss.backward()
        # clip grads so they dont explode when fine tuning all layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t_loss    += loss.item() * imgs.size(0)
        t_correct += (pred == lbls).sum().item()

    t_loss /= len(train_ds)
    t_acc   = t_correct / len(train_ds)

    # -- validate --
    model.eval()
    v_loss    = 0.0
    v_correct = 0

    with torch.no_grad():
        for imgs, lbls in tqdm(val_dl, desc="val", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss = loss_fn(out, lbls)
            pred = out.argmax(dim=1)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (pred == lbls).sum().item()

    v_loss /= len(val_ds)
    v_acc   = v_correct / len(val_ds)

    log["tloss"].append(t_loss)
    log["tacc"].append(t_acc)
    log["vloss"].append(v_loss)
    log["vacc"].append(v_acc)

    print(f"  train  loss={t_loss:.4f}  acc={t_acc:.4f}")
    print(f"  val    loss={v_loss:.4f}  acc={v_acc:.4f}")

    scheduler.step()

    # save best model / early stopping
    if v_acc > best_acc:
        best_acc   = v_acc
        best_wts   = copy.deepcopy(model.state_dict())
        no_improve = 0
        print(f"  saved best model (val_acc={best_acc:.4f})")
    else:
        no_improve += 1
        print(f"  no improvement ({no_improve}/{stop_after})")
        if no_improve >= stop_after:
            print("early stopping")
            break

# load back the best weights
model.load_state_dict(best_wts)
n_ep = len(log["tloss"])
print(f"\ntraining done. best val acc: {best_acc:.4f}")


# ------- plot training curves -------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, n_ep+1), log["tloss"], marker="o", label="train")
ax1.plot(range(1, n_ep+1), log["vloss"], marker="o", label="val")
ax1.set_title("Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, n_ep+1), log["tacc"], marker="o", label="train")
ax2.plot(range(1, n_ep+1), log["vacc"], marker="o", label="val")
ax2.set_title("Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("train_val_plot.png", dpi=150)
plt.show()


# ------- test evaluation -------
model.eval()
all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for imgs, lbls in tqdm(test_dl, desc="testing"):
        imgs = imgs.to(device)
        out  = model(imgs)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)
        pred  = probs.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbls.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc = (all_preds == all_labels).mean()
print(f"\ntest accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nclassification report:")
print(classification_report(all_labels, all_preds, target_names=labels))

# auc roc
bin_labels = label_binarize(all_labels, classes=list(range(n_classes)))
macro_auc  = roc_auc_score(bin_labels, all_probs, multi_class="ovr", average="macro")

print("per class AUC:")
for i, cls in enumerate(labels):
    sc = roc_auc_score(bin_labels[:, i], all_probs[:, i])
    print(f"  {cls}: {sc:.4f}")
print(f"macro AUC: {macro_auc:.4f}")


# ------- confusion matrix -------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", linewidths=0.5)
plt.title("Confusion Matrix - Test Set", fontweight="bold")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# ------- roc curves -------
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#e63946", "#2a9d8f", "#e9c46a", "#264653"]

for i, (cls, col) in enumerate(zip(labels, colors)):
    fpr, tpr, _ = roc_curve(bin_labels[:, i], all_probs[:, i])
    score       = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC={score:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - One vs Rest")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()


# ------- save model -------
torch.save({
    "model_state": best_wts,
    "optimizer":   optimizer.state_dict(),
    "epochs":      n_ep,
    "val_acc":     best_acc,
    "classes":     labels
}, out_model)

print(f"\nmodel saved to {out_model}")
print(f"test acc   : {acc*100:.2f}%")
print(f"macro AUC  : {macro_auc:.4f}")
print(f"epochs ran : {n_ep}")
