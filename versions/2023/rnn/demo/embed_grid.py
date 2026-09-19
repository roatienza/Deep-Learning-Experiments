"""Embed the rendered 1x4 sample-image grid into rnn_mnist.ipynb.

Renders the section 7.5 grid (4 random test images, GT vs prediction,
from the best checkpoint) with matplotlib, saves it as a PNG, and
injects it into the notebook as a base64 image output on the section
7.5 code cell so the figure is visible without re-running the cell.

Run:  python embed_grid.py
"""
import base64
import io
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms

NB_PATH = os.path.join(HERE, 'rnn_mnist.ipynb')
CKPT_PATH = os.path.join(HERE, 'rnn_mnist_best.pt')

# ------------------------------------------------------------------
# Reproduce the section 7.5 figure (same code path as the notebook)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = datasets.MNIST(
    root='data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))


class RNN(nn.Module):
    # Classify a 28x28 image as a sequence of 28 rows of 28 pixels.
    def __init__(self, input_size=28, hidden_size=256, num_layers=1,
                 num_classes=10):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        y, h = self.rnn(x)
        return self.fc(y[:, -1, :]), h


model = RNN()
checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

random.seed(42)  # deterministic selection for a stable embedded figure
indices = random.sample(range(len(test_dataset)), 4)
images = [test_dataset[i][0] for i in indices]
labels = [test_dataset[i][1] for i in indices]

with torch.no_grad():
    outputs, _ = model(torch.stack(images).to(device))
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

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
plt.close(fig)
png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
print(f"grid: GT {labels}  Pred {predicted}  "
      f"mismatches {sum(p != l for p, l in zip(predicted, labels))}  "
      f"png {len(png_b64)} b64 chars")

# ------------------------------------------------------------------
# Inject into the notebook on the section 7.5 code cell
nb = json.load(open(NB_PATH))
target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'Sample predictions from checkpoint' in ''.join(c['source']):
        target = i
        break
if target is None:
    raise SystemExit('section 7.5 code cell not found')

nb['cells'][target]['outputs'] = [{
    'output_type': 'display_data',
    'data': {'image/png': png_b64},
    'metadata': {},
}]
nb['cells'][target]['execution_count'] = 1

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
print(f"embedded grid into cell {target} of {NB_PATH}")
