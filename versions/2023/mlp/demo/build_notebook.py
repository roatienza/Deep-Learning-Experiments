"""Build the improved mlp_mnist.ipynb.

Run:  python build_notebook.py
Writes mlp_mnist.ipynb next to this script.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

# ---------------------------------------------------------------- cell 0
cells.append(md("""\
## MNIST Image Classification with an MLP Model

A Multilayer Perceptron (MLP) is the simplest form of neural network:
fully-connected layers stacked with nonlinear activations. Despite its
simplicity, an MLP is a strong baseline for image classification when the
input is flattened into a vector, and it is the building block that later
architectures (CNNs, Transformers) extend.

In this notebook we:

1. Load and prepare the MNIST dataset (train / validation / test splits).
2. Visualize a few samples.
3. Define a small MLP (784 -> 256 -> 128 -> 10) in PyTorch.
4. Train it with Adam, weight decay, and a learning-rate scheduler.
5. Evaluate it: overall accuracy, per-class accuracy, and a confusion matrix.
6. Save the best checkpoint and inspect a few predictions.

Expected result: **~98% test accuracy** for the MLP on MNIST.
"""))

# ---------------------------------------------------------------- cell 1
cells.append(md("""\
### 1. Build the dataset and dataloaders

We split the official 60,000-image MNIST training set into a 55,000 training
portion and a 5,000 validation portion. The validation set is used **only**
to monitor generalization during training and to pick the best checkpoint;
the official 10,000-image test set is touched once, at the very end.

Images are converted to tensors and normalized with mean=0.5, std=0.5, which
maps the original [0, 1] pixel range to [-1, 1]. Normalization keeps the
input scale consistent with the weight initialization and typically speeds
up convergence.
"""))

cells.append(code("""\
import torch
import torchvision
from torchvision import transforms

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Transform: tensor + normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load the full MNIST train set, then split into train / validation
full_train = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(
    full_train,
    [55000, 5000],
    generator=torch.Generator().manual_seed(SEED),
)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

print(f"Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}  |  "
      f"Test: {len(test_dataset):,}")

# Dataloaders
batch_size = 128
num_workers = 2
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
"""))

# ---------------------------------------------------------------- cell 2
cells.append(md("""\
### 2. Visualize samples from the dataset

A quick sanity check: 16 random training images with their labels.
MNIST digits are 28x28 grayscale, centered in the frame.
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
import numpy as np

indices = np.random.choice(len(train_dataset), size=16, replace=False)
images = [train_dataset[i][0] for i in indices]
labels = [train_dataset[i][1] for i in indices]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f"Label: {labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------- cell 3
cells.append(md("""\
### 3. Build the MLP model

Architecture:

```
input (1 x 28 x 28)
  -> flatten -> 784
  -> Linear(784, 256) -> ReLU
  -> Linear(256, 128) -> ReLU
  -> Linear(128, 10)   -> logits
```

Notes:

* **Two hidden layers** (256, 128) give the model enough capacity to learn
  the non-linear decision boundaries between digits, while staying small
  enough to train in seconds on a GPU.
* **ReLU** is the standard hidden activation: cheap, and it avoids the
  vanishing-gradient problem of sigmoid/tanh in deep stacks.
* The **output layer has no activation** — we return raw logits because
  `nn.CrossEntropyLoss` applies log-softmax internally.
* `x.view(x.size(0), -1)` flattens the image; the batch dimension is
  preserved so the model works for any batch size.
"""))

cells.append(code("""\
import torch.nn as nn


class MLP(nn.Module):
    # A simple feed-forward network for MNIST.

    def __init__(self, input_size=28 * 28, hidden_sizes=(256, 128),
                 num_classes=10):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten (B, 1, 28, 28) -> (B, 784)
        return self.net(x)


model = MLP()
x = torch.randn(64, 1, 28, 28)
print(model)
print("output shape:", model(x).shape)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
"""))

# ---------------------------------------------------------------- cell 4
cells.append(md("""\
### 4. Loss function and optimizer

* **Loss:** `CrossEntropyLoss` (log-softmax + NLL) — the standard choice for
  single-label classification.
* **Optimizer:** Adam with `lr=1e-3` and **weight decay `1e-4`** (L2
  regularization). Weight decay is the most effective simple regularizer for
  an MLP of this size; it noticeably improves test accuracy over an
  unregularized run.
* **Scheduler:** `CosineAnnealingLR` decays the learning rate smoothly to
  ~0 over the training run, which helps the model settle into a sharper
  minimum in the final epochs.
"""))

cells.append(code("""\
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
"""))

# ---------------------------------------------------------------- cell 5
cells.append(md("""\
### 5. Train the model

The training loop, per epoch:

1. Forward pass on a batch, compute the loss.
2. Backward pass, then an optimizer step (gradients zeroed first).
3. After the epoch: evaluate on the validation set, log both accuracies,
   step the scheduler, and **save a checkpoint if validation accuracy
   improved** (best-model selection).

We train for 10 epochs — plenty for this model to converge.
"""))

cells.append(code("""\
from tqdm import tqdm


def evaluate(model, loader, device):
    # Return (accuracy, total_loss) over a dataloader, no gradient tracking.
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / total


num_epochs = 10
best_val_acc = 0.0
history = []

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}",
                               leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_dataset)

    val_acc, val_loss = evaluate(model, val_loader, device)
    scheduler.step()
    history.append((epoch, train_loss, val_acc, val_loss))

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'mlp_mnist_best.pt')

    print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | "
          f"val acc {val_acc:.4f} | val loss {val_loss:.4f} | "
          f"best {best_val_acc:.4f}{' *' if improved else ''}")
"""))

# ---------------------------------------------------------------- cell 6
cells.append(md("""\
### 6. Evaluate the model

We load the **best checkpoint** (selected on validation accuracy) and report:

* overall test accuracy,
* per-class accuracy (useful for spotting which digits are confused),
* a confusion matrix over the 10,000 test images.
"""))

cells.append(code("""\
# Load the best checkpoint
checkpoint = torch.load('mlp_mnist_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
      f"(val acc {checkpoint['val_acc']:.4f})")

# Overall test accuracy
test_acc, test_loss = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")
"""))

# ---------------------------------------------------------------- cell 7
cells.append(md("""\
### 7. Per-class accuracy and confusion matrix
"""))

cells.append(code("""\
from collections import defaultdict

model.eval()
tp = defaultdict(int)
fn = defaultdict(int)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        for true, pred in zip(labels.tolist(), predicted.tolist()):
            if true == pred:
                tp[true] += 1
            else:
                fn[true] += 1

print(f"{'digit':>6} | {'correct':>8} | {'wrong':>6} | {'acc':>7}")
for d in range(10):
    n = tp[d] + fn[d]
    print(f"{d:>6} | {tp[d]:>8} | {fn[d]:>6} | {tp[d] / n:7.2%}")
"""))

cells.append(code("""\
import numpy as np

# Confusion matrix
model.eval()
cm = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        for true, pred in zip(labels.tolist(), predicted.tolist()):
            cm[true, pred] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10), [str(i) for i in range(10)])
ax.set_yticks(range(10), [str(i) for i in range(10)])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix (MNIST test set)')
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=7,
                    color='white' if cm[i, j] > cm[i].max() * 0.5 else 'black')
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------- cell 8
cells.append(md("""\
### 8. Predict on sample images

A few random test images with their true labels and the model's predictions.
"""))

cells.append(code("""\
import random

model.eval()
indices = random.sample(range(len(test_dataset)), 9)
images = [test_dataset[i][0] for i in indices]
labels = [test_dataset[i][1] for i in indices]

with torch.no_grad():
    outputs = model(torch.stack(images).to(device))
    predicted = outputs.argmax(dim=1).cpu().tolist()

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ok = predicted[i] == labels[i]
    ax.set_title(f"Label: {labels[i]}  Pred: {predicted[i]}"
                 f"  {'OK' if ok else 'X'}",
                 color='green' if ok else 'red')
    ax.axis('off')
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------- cell 9
cells.append(md("""\
### Summary

| Item | Value |
|---|---|
| Architecture | 784 -> 256 -> 128 -> 10, ReLU |
| Parameters | ~203k |
| Optimizer | Adam, lr=1e-3, weight decay 1e-4 |
| Scheduler | CosineAnnealingLR (T_max = epochs) |
| Training | 10 epochs, batch size 128 |
| Test accuracy | ~98% (best checkpoint) |

The MLP is a strong baseline: a CNN reaches ~99.2% on the same data, and a
Transformer ~99.4%, so the gap to close is small — but the MLP gets you most
of the way there with a few hundred thousand parameters and no convolutions.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(HERE, 'mlp_mnist.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
print(f"wrote {out} with {len(cells)} cells")
