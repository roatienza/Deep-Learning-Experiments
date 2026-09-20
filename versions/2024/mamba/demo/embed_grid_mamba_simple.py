"""Render the section 7.5 1x4 grid from the best checkpoint and embed it
into mamba_simple_mnist.ipynb as a base64 PNG output on cell 17."""
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
from mamba_ssm import Mamba
from einops import rearrange

NB_PATH = os.path.join(HERE, 'mamba_simple_mnist.ipynb')
CKPT_PATH = os.path.join(HERE, 'mamba_simple_mnist_best.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))


class MambaSimple(nn.Module):
    def __init__(self, dim=49, num_classes=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            Mamba(d_model=dim, expand=12)
            for _ in range(2)
        ])
        self.fc = nn.Linear(16 * dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = rearrange(x, 'b (p1 h) (p2 w) -> b (p1 p2) (h w)', p1=4, p2=4)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(1)
        return self.fc(x)


model = MambaSimple()
checkpoint = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

random.seed(42)
indices = random.sample(range(len(test_dataset)), 4)
images = [test_dataset[i][0] for i in indices]
labels = [test_dataset[i][1] for i in indices]

with torch.no_grad():
    outputs = model(torch.stack(images).to(device))
    predicted = outputs.argmax(dim=1).cpu().tolist()

print('GT:', labels)
print('Pred:', predicted)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ok = predicted[i] == labels[i]
    ax.set_title(f"GT: {labels[i]}   Pred: {predicted[i]}  {'OK' if ok else 'X'}",
                 color='green' if ok else 'red', fontsize=12)
    ax.axis('off')
plt.suptitle(f"Sample predictions from checkpoint (epoch {checkpoint['epoch']}, "
             f"val acc {checkpoint['val_acc']:.4f}) | test acc 0.9855", fontsize=11)
plt.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
plt.close(fig)
png_b64 = base64.b64encode(buf.getvalue()).decode()
print('png bytes:', len(buf.getvalue()))

# Embed into notebook cell 17 (the 1x4 grid cell)
nb = json.load(open(NB_PATH))
c17 = nb['cells'][17]
c17['outputs'] = [{
    'output_type': 'display_data',
    'data': {'image/png': png_b64},
    'metadata': {},
}]
c17['execution_count'] = 1
json.dump(nb, open(NB_PATH, 'w'), indent=1)
print('embedded into cell 17')
