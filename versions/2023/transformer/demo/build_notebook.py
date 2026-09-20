"""Build the improved transformer_mnist.ipynb (18 cells).

Run:  python build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, 'transformer_mnist.ipynb')

CELLS = []


def md(src):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(keepends=True)})


def code(src):
    CELLS.append({'cell_type': 'code', 'execution_count': None,
                  'metadata': {}, 'outputs': [], 'source': src.splitlines(keepends=True)})


# ------------------------------------------------------------------ 0 intro
md("""## MNIST Image Classification with a Transformer Model

A Transformer is a sequence model built on self-attention. For images we
treat the picture as a sequence of *patches*: a 28x28 image is split into a
4x4 grid of 7x7 patches (16 tokens, each 49 pixels). A single
`nn.TransformerEncoderLayer` lets every patch attend to every other patch,
so the model can reason about global structure (the whole digit at once)
rather than local filters as in a CNN. The 16 patch embeddings are then
flattened (16 x 49 = 784) and a linear head maps them to 10 class logits.

In this notebook we:

1. Load and prepare the MNIST dataset (train / validation / test splits).
2. Visualize a few samples.
3. Define a small patch-based Transformer in PyTorch.
4. Train it with Adam, weight decay, and a learning-rate scheduler.
5. Evaluate it: overall accuracy, per-class accuracy, a confusion matrix,
   and a 1x4 grid of sample images with their ground-truth and predicted
   labels.

**Architecture.** 4x4 patches of 7x7 pixels -> 16 tokens of dimension 49 ->
one `TransformerEncoderLayer` (d_model=49, nhead=7) -> flatten (16 x 49) ->
`Linear(784, 10)`. No learned positional embeddings are added; the patch
order (row-major) is the only position signal, which is enough for MNIST.

Expected result: **~98% test accuracy** for this small Transformer on MNIST.
""")

# ------------------------------------------------------------------ 1 data md
md("""### 1. Build the dataset and dataloaders

We split the official 60,000-image MNIST training set into a 55,000 training
portion and a 5,000 validation portion. The validation set is used **only**
to monitor generalization during training and to pick the best checkpoint;
the official 10,000-image test set is touched once, at the very end.

Images are converted to tensors and normalized with mean=0.5, std=0.5, which
maps the original [0, 1] pixel range to [-1, 1].
""")

# ------------------------------------------------------------------ 2 data code
code("""import torch
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
""")

# ------------------------------------------------------------------ 3 viz md
md("""### 2. Visualize samples from the dataset

A quick sanity check: 16 random training images with their labels.
MNIST digits are 28x28 grayscale, centered in the frame.
""")

# ------------------------------------------------------------------ 4 viz code
code("""import matplotlib.pyplot as plt
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
""")

# ------------------------------------------------------------------ 5 model md
md("""### 3. Build the Transformer model

Architecture:

```
input (1 x 28 x 28)
  -> patchify 4x4 (each patch 7x7) -> (B, 16, 49)
  -> TransformerEncoderLayer(d_model=49, nhead=7, batch_first=True)
  -> flatten (16 x 49 = 784)
  -> Linear(784, 10) -> logits
```

Notes:

* **Patchify** with `einops.rearrange`: `'b (p1 h) (p2 w) -> b (p1 p2) (h w)'`
  with `p1=p2=4` splits the 28x28 image into a 4x4 grid of 7x7 patches,
  producing 16 tokens of 49 pixels each.
* **One encoder layer** with `d_model=49` and `nhead=7` (49 = 7 x 7, so the
  model dimension divides evenly by the head count). Every patch attends to
  every other patch, which gives the model a global view of the digit.
* **No positional embeddings**: the row-major patch order is the only
  position signal. For MNIST this is enough; for harder datasets you would
  add learned positional encodings.
* The **output layer has no activation** -- we return raw logits because
  `nn.CrossEntropyLoss` applies log-softmax internally.
""")

# ------------------------------------------------------------------ 6 model code
code("""import torch
import torch.nn as nn
from einops import rearrange


class Transformer(nn.Module):
    # Patch-based Transformer for MNIST: 4x4 patches -> 1 encoder layer -> head.

    def __init__(self, dim=49, num_classes=10):
        super().__init__()
        self.xformer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=7,
            batch_first=True,
        )
        self.fc = nn.Linear(16 * dim, num_classes)

    def forward(self, x):
        # (B, 1, 28, 28) -> (B, 28, 28)
        x = x.squeeze(1)
        # (B, 28, 28) -> (B, 16, 49): 4x4 grid of 7x7 patches
        x = rearrange(x, 'b (p1 h) (p2 w) -> b (p1 p2) (h w)', p1=4, p2=4)
        x = self.xformer(x)
        x = x.flatten(1)
        return self.fc(x)


model = Transformer()
x = torch.randn(64, 1, 28, 28)
print(model)
print("output shape:", model(x).shape)

num_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {num_params:,}")
""")

# ------------------------------------------------------------------ 7 loss md
md("""### 4. Loss function and optimizer

* **Loss:** `CrossEntropyLoss` (log-softmax + NLL) -- the standard choice for
  single-label classification.
* **Optimizer:** Adam with `lr=1e-3` and **weight decay `1e-4`** (L2
  regularization).
* **Scheduler:** cosine annealing over the whole run -- the learning rate
  starts at 1e-3 and decays smoothly to ~0, which gives fast early progress
  and fine-tuned late training.
""")

# ------------------------------------------------------------------ 8 loss code
code("""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs)
print(f"device: {device}")
""")

# ------------------------------------------------------------------ 9 train md
md("""### 5. Train the model

Each epoch: train on the 55,000-image train split, evaluate on the 5,000-
image validation split, and save the checkpoint with the best validation
accuracy. The test set is **not** used here.
""")

# ------------------------------------------------------------------ 10 train code
code("""def evaluate(model, loader):
    # Return (accuracy, avg_loss) over a dataloader, no gradients.
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


best_val_acc = 0.0
history = []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
    train_loss = total_loss / len(train_dataset)

    val_acc, val_loss = evaluate(model, val_loader)
    scheduler.step()
    history.append((epoch, train_loss, val_acc))
    print(f"Epoch {epoch:2d}/{num_epochs}  "
          f"train loss {train_loss:.4f}  val acc {val_acc:.4f}  "
          f"lr {scheduler.get_last_lr()[0]:.5f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, 'transformer_mnist_best.pt')

print(f"\\nbest validation accuracy: {best_val_acc:.4f}")
""")

# ------------------------------------------------------------------ 11 eval md
md("""### 6. Evaluate the model

We load the **best checkpoint** (selected on validation accuracy) and report:

* overall test accuracy,
* per-class accuracy (useful for spotting which digits are confused),
* a confusion matrix over the 10,000 test images.
""")

# ------------------------------------------------------------------ 12 eval code
code("""import numpy as np

checkpoint = torch.load('transformer_mnist_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
      f"(val acc {checkpoint['val_acc']:.4f})")

# Overall test accuracy
test_acc, test_loss = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}   Test Loss: {test_loss:.4f}")

# Per-class accuracy + confusion matrix
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

all_preds, all_labels = np.array(all_preds), np.array(all_labels)
print("\\nPer-class accuracy:")
for d in range(10):
    mask = all_labels == d
    acc = (all_preds[mask] == d).mean()
    print(f"  digit {d}: {acc:.4f}  (n={mask.sum()})")

cm = np.zeros((10, 10), dtype=int)
for p, t in zip(all_preds, all_labels):
    cm[t, p] += 1

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix (test set)')
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    fontsize=7,
                    color='white' if cm[i, j] > cm[i].max() * 0.5 else 'black')
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
""")

# ------------------------------------------------------------------ 13 3x3 md
md("""### 7. Predict on sample images

A 3x3 grid of random test images with the model's prediction next to the
ground-truth label.
""")

# ------------------------------------------------------------------ 14 3x3 code
code("""import random

model.eval()
indices = random.sample(range(len(test_dataset)), 9)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    image, label = test_dataset[indices[i]]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"GT: {label}   Pred: {pred}",
                 color='green' if pred == label else 'red')
    ax.axis('off')
plt.tight_layout()
plt.show()
""")

# ------------------------------------------------------------------ 15 7.5 md
md("""### 7.5. Sample images from the checkpoint: ground truth vs prediction

This cell is **self-contained**: it reloads `transformer_mnist_best.pt` from
disk and runs four random test images through it, so you can re-run it at any
time (or in a fresh kernel) to see what the trained checkpoint actually
predicts. Each panel shows **GT** (the ground-truth digit) and **Pred** (the
model's argmax prediction); the title is green when they match and red when
they do not.
""")

# ------------------------------------------------------------------ 16 7.5 code
code("""import random

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))


class Transformer(nn.Module):
    def __init__(self, dim=49, num_classes=10):
        super().__init__()
        self.xformer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=7, batch_first=True)
        self.fc = nn.Linear(16 * dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = rearrange(x, 'b (p1 h) (p2 w) -> b (p1 p2) (h w)', p1=4, p2=4)
        x = self.xformer(x)
        x = x.flatten(1)
        return self.fc(x)


model = Transformer()
checkpoint = torch.load('transformer_mnist_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

indices = random.sample(range(len(test_dataset)), 4)
images = [test_dataset[i][0] for i in indices]
labels = [test_dataset[i][1] for i in indices]

with torch.no_grad():
    outputs = model(torch.stack(images).to(device))
    predicted = outputs.argmax(dim=1).cpu().tolist()

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ok = predicted[i] == labels[i]
    ax.set_title(f"GT: {labels[i]}   Pred: {predicted[i]}  {'OK' if ok else 'X'}",
                 color='green' if ok else 'red', fontsize=12)
    ax.axis('off')
plt.suptitle(f"Sample predictions from checkpoint (epoch {checkpoint['epoch']}, "
             f"val acc {checkpoint['val_acc']:.4f})", fontsize=11)
plt.tight_layout()
plt.show()
""")

# ------------------------------------------------------------------ 17 summary
md("""### 8. Summary

| Item | Value |
|---|---|
| Architecture | Patch Transformer: 4x4 patches (7x7) -> 1 encoder layer (d_model 49, nhead 7) -> Linear(784, 10) |
| Parameters | ~20k |
| Training | Adam (lr 1e-3, wd 1e-4) + cosine annealing, 10 epochs |
| Data | 55k train / 5k val / 10k test |
| Best checkpoint | selected on validation accuracy |

The Transformer treats the image as a 16-token sequence and lets every patch
attend to every other patch, which gives it a global view of the digit in a
single layer. Section 7.5 lets you visually inspect what the checkpoint
predicts on fresh test images.
""")

nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print(f"wrote {NB_PATH} with {len(CELLS)} cells")
