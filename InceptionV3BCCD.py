"""
Blood cell classification — BCCD dataset, 4 cell types
Fine-tuning InceptionV3 end-to-end. Running on Colab T4.

Went with inception mainly because the cell images have a lot of fine-grained
texture that the wider receptive fields handle better in my experience.
Also 299x299 gives more pixel info for small structures like platelets.
"""

import os
import copy
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────
# keeping these in one dict so they're easy to tweak without hunting
# through the whole file

CFG = {
    "root":      "/content/data",
    "img_px":    299,           # inception requires exactly 299x299
    "n_cls":     4,
    "epochs":    30,
    "lr":        1e-5,          # full fine-tune so keeping this tight to avoid blowing up early layers
    "bs":        16,            # 32 caused OOM with all layers unfrozen on T4
    "tr_frac":   0.70,
    "val_frac":  0.15,
    # test gets the remaining 0.15
    "patience":  7,
    "seed":      42,
    "save_as":   "model_v2.pth",
}

torch.manual_seed(CFG["seed"])
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", gpu)
if gpu.type == "cuda":
    print("  gpu:", torch.cuda.get_device_name(0))


# ── transforms ────────────────────────────────────────────────────────────────
# training gets the full augmentation stack; val/test just get resized and
# normalised — don't want randomness affecting the metric numbers

# ImageNet stats even for microscopy because the pretrained weights expect them
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
PX   = CFG["img_px"]

augment_tf = transforms.Compose([
    transforms.Resize((PX, PX)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    # cells can vary in staining intensity across slides so color jitter helps
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((PX, PX)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ── dataset ───────────────────────────────────────────────────────────────────
# custom dataset so we can apply different transforms to each split
# while still using the same underlying files from ImageFolder

class CellDataset(Dataset):
    """Wraps a random_split subset and applies its own transform."""
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf     = tf

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        real_idx      = self.subset.indices[i]
        path, label   = self.subset.dataset.samples[real_idx]
        img           = Image.open(path).convert("RGB")
        return self.tf(img), label


# ── load and split ────────────────────────────────────────────────────────────

from torchvision import datasets as tvdatasets

raw = tvdatasets.ImageFolder(root=CFG["root"], transform=augment_tf)
cell_types = raw.classes
print("classes:", cell_types)
print("total  :", len(raw))

total   = len(raw)
n_tr    = int(total * CFG["tr_frac"])
n_val   = int(total * CFG["val_frac"])
n_te    = total - n_tr - n_val

# fixed generator so the split is reproducible run-to-run
rng = torch.Generator().manual_seed(CFG["seed"])
tr_sub, val_sub, te_sub = random_split(raw, [n_tr, n_val, n_te], generator=rng)

train_set = CellDataset(tr_sub,  augment_tf)
val_set   = CellDataset(val_sub, eval_tf)
test_set  = CellDataset(te_sub,  eval_tf)

train_loader = DataLoader(train_set, batch_size=CFG["bs"], shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=CFG["bs"], shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=CFG["bs"], shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")


# ── model ─────────────────────────────────────────────────────────────────────

net = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
# swap both heads — main and the auxiliary branch used during training
net.fc           = nn.Linear(net.fc.in_features, CFG["n_cls"])
net.AuxLogits.fc = nn.Linear(net.AuxLogits.fc.in_features, CFG["n_cls"])

# unfreeze everything — full fine-tune gives better results here than
# head-only, even though it's slower and needs a lower LR
for p in net.parameters():
    p.requires_grad = True

net = net.to(gpu)
print(f"param count: {sum(p.numel() for p in net.parameters()):,}")


# ── loss / optimiser / scheduler ──────────────────────────────────────────────

# label smoothing: stops the model from becoming overconfident,
# which was visibly happening in early runs without it
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
opt     = optim.Adam(net.parameters(), lr=CFG["lr"], weight_decay=1e-4)
# cosine annealing lets the LR decay smoothly rather than step drops
# which felt unstable when fine-tuning all 24M params at once
sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-7)


# ── training helpers ──────────────────────────────────────────────────────────

class EarlyStopper:
    """Tracks val accuracy and signals when to quit."""
    def __init__(self, patience):
        self.patience  = patience
        self.strikes   = 0
        self.best_acc  = 0.0
        self.best_wts  = None

    def update(self, val_acc, model):
        """Returns True if training should stop."""
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_wts = copy.deepcopy(model.state_dict())
            self.strikes  = 0
            return False
        self.strikes += 1
        return self.strikes >= self.patience


def train_one_epoch(model, loader):
    model.train()
    running_loss, correct = 0.0, 0

    for imgs, lbls in tqdm(loader, desc="train", leave=False):
        imgs, lbls = imgs.to(gpu), lbls.to(gpu)
        opt.zero_grad()

        out = model(imgs)
        # inception gives (main_logits, aux_logits) during training
        if isinstance(out, tuple):
            main_out, aux_out = out
            loss  = loss_fn(main_out, lbls) + 0.4 * loss_fn(aux_out, lbls)
            preds = main_out.argmax(dim=1)
        else:
            loss  = loss_fn(out, lbls)
            preds = out.argmax(dim=1)

        loss.backward()
        # grad clipping — without this, full fine-tune occasionally spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item() * imgs.size(0)
        correct      += (preds == lbls).sum().item()

    n = len(loader.dataset)
    return running_loss / n, correct / n


def evaluate(model, loader):
    model.eval()
    running_loss, correct = 0.0, 0

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="val", leave=False):
            imgs, lbls = imgs.to(gpu), lbls.to(gpu)
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            loss  = loss_fn(out, lbls)
            preds = out.argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            correct      += (preds == lbls).sum().item()

    n = len(loader.dataset)
    return running_loss / n, correct / n


# ── training loop ─────────────────────────────────────────────────────────────

history = {"tr_loss": [], "tr_acc": [], "val_loss": [], "val_acc": []}
stopper = EarlyStopper(patience=CFG["patience"])

for ep in range(1, CFG["epochs"] + 1):
    cur_lr = sched.get_last_lr()[0]
    print(f"\n[{ep}/{CFG['epochs']}]  lr={cur_lr:.2e}")
    print("-" * 38)

    tr_loss, tr_acc   = train_one_epoch(net, train_loader)
    val_loss, val_acc = evaluate(net, val_loader)

    history["tr_loss"].append(tr_loss)
    history["tr_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"  train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
    print(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}")

    sched.step()

    should_stop = stopper.update(val_acc, net)
    if stopper.strikes == 0:
        print(f"  ↑ new best  val_acc={stopper.best_acc:.4f}")
    else:
        print(f"  no gain ({stopper.strikes}/{CFG['patience']})")
    if should_stop:
        print("early stopping triggered")
        break

net.load_state_dict(stopper.best_wts)
n_ep = len(history["tr_loss"])
print(f"\nfinished. best val acc: {stopper.best_acc:.4f}  ({n_ep} epochs ran)")


# ── training curves ───────────────────────────────────────────────────────────

xs = range(1, n_ep + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(xs, history["tr_loss"], "o-", label="train")
ax1.plot(xs, history["val_loss"], "o-", label="val")
ax1.set(title="Loss per Epoch", xlabel="Epoch", ylabel="Loss")
ax1.legend(); ax1.grid(True)

ax2.plot(xs, history["tr_acc"], "o-", label="train")
ax2.plot(xs, history["val_acc"], "o-", label="val")
ax2.set(title="Accuracy per Epoch", xlabel="Epoch", ylabel="Accuracy")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("train_val_plot.png", dpi=150)
plt.show()


# ── test set evaluation ───────────────────────────────────────────────────────

net.eval()
all_preds, all_lbls, all_probs = [], [], []

with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="testing"):
        out   = net(imgs.to(gpu))
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_lbls.extend(lbls.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_lbls  = np.array(all_lbls)
all_probs = np.array(all_probs)

test_acc = (all_preds == all_lbls).mean()
print(f"\ntest acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\nper-class breakdown:")
print(classification_report(all_lbls, all_preds, target_names=cell_types))

# AUC-ROC
bin_true  = label_binarize(all_lbls, classes=list(range(CFG["n_cls"])))
macro_auc = roc_auc_score(bin_true, all_probs, multi_class="ovr", average="macro")

print("per-class AUC:")
for i, cname in enumerate(cell_types):
    sc = roc_auc_score(bin_true[:, i], all_probs[:, i])
    print(f"  {cname}: {sc:.4f}")
print(f"macro AUC: {macro_auc:.4f}")


# ── confusion matrix ──────────────────────────────────────────────────────────

cm = confusion_matrix(all_lbls, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=cell_types, yticklabels=cell_types,
            cmap="Blues", linewidths=0.5)
plt.title("Confusion Matrix — Test Set", fontweight="bold")
plt.ylabel("True label")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# ── ROC curves ────────────────────────────────────────────────────────────────

palette = ["#e63946", "#2a9d8f", "#e9c46a", "#264653"]
fig, ax = plt.subplots(figsize=(9, 7))

for i, (cname, col) in enumerate(zip(cell_types, palette)):
    fpr, tpr, _ = roc_curve(bin_true[:, i], all_probs[:, i])
    score       = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{cname}  (AUC={score:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curves — One vs Rest")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()


# ── save ──────────────────────────────────────────────────────────────────────

torch.save({
    "weights":    stopper.best_wts,
    "optimizer":  opt.state_dict(),
    "epochs_ran": n_ep,
    "val_acc":    stopper.best_acc,
    "test_acc":   float(test_acc),
    "classes":    cell_types,
}, CFG["save_as"])

print(f"\nsaved → {CFG['save_as']}")
print(f"test acc : {test_acc*100:.2f}%")
print(f"macro AUC: {macro_auc:.4f}")
print(f"epochs   : {n_ep}")
