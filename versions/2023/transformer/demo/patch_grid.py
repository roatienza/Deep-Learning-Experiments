"""Add a 1x4 GT-vs-prediction sample grid (section 7.5) to transformer_cifar10.ipynb.
Inserts a markdown cell + a code cell after cell 15 (the training cell).
The code cell is self-contained: it finds the best checkpoint, loads it,
and renders the grid.
"""
import json

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

MD = """### 7.5. Sample images from the checkpoint: ground truth vs prediction

This cell loads the best checkpoint saved during training and shows a **1x4
grid** of random test images. Each panel displays the image with its
**ground-truth (GT)** label and the model's **predicted (Pred)** label.
Titles are green when the prediction matches the ground truth and red when
it does not. Red panels typically expose the classic CIFAR confusions
(e.g. 3/5, 8/9, 0/4), which are the same confusions you see in the
per-class accuracy table and the confusion matrix in section 7.
"""

CODE = '''import os
import random
import torch
import matplotlib.pyplot as plt
from einops import rearrange

# Find the best checkpoint saved by the ModelCheckpoint callback
ckpt_path = None
for root, dirs, files in os.walk("lightning_logs"):
    for f in files:
        if f == "transformer_cifar10_best.ckpt":
            ckpt_path = os.path.join(root, f)
assert ckpt_path is not None, "No checkpoint found. Run the training cell first."
print(f"Loading checkpoint: {ckpt_path}")

# Load the model from the checkpoint
best_model = LitTransformer.load_from_checkpoint(ckpt_path)
best_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model.to(device)

# Load the test set (raw images, not patches)
test_set = torchvision.datasets.CIFAR10(root="~/data", train=False,
                                        download=True,
                                        transform=torchvision.transforms.ToTensor())

# Pick 4 random test images
random.seed(42)
idxs = random.sample(range(len(test_set)), 4)
images = torch.stack([test_set[i][0] for i in idxs]).to(device)
labels = torch.tensor([test_set[i][1] for i in idxs]).to(device)

# The model expects patches: (B, C, H, W) -> (B, N, patch_dim)
with torch.no_grad():
    x = rearrange(images, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=4, p2=4)
    preds = best_model(x).argmax(dim=1)

# Render the 1x4 grid
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, img, gt, pred in zip(axes, images.cpu(), labels.cpu(), preds.cpu()):
    ax.imshow(img.clamp(0, 1))
    ok = gt == pred
    ax.set_title(f"GT {gt} | Pred {pred}", color="green" if ok else "red",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
plt.suptitle(f"Transformer CIFAR10 checkpoint", fontsize=10)
plt.tight_layout()
plt.show()
'''

# Insert after cell 15 (index 15), before the empty cell 16
md_cell = {'cell_type': 'markdown', 'metadata': {}, 'source': MD.splitlines(keepends=True)}
code_cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {},
             'outputs': [], 'source': CODE.splitlines(keepends=True)}
nb['cells'].insert(16, md_cell)
nb['cells'].insert(17, code_cell)

json.dump(nb, open(p, 'w'), indent=1)
print('INSERTED 7.5, now', len(nb['cells']), 'cells')
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first = src.strip().splitlines()[0][:80] if src.strip() else '(empty)'
    print(i, c['cell_type'], '|', first)
