"""Update cell 16 (section 7.5) of transformer_mnist.ipynb to the new model
architecture (positional embedding + patch projection) and seed the sample
selection for a stable embedded figure."""
import json

NB = 'transformer_mnist.ipynb'
nb = json.load(open(NB))

new16 = """import random

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
    def __init__(self, dim=49, num_classes=10, num_patches=16):
        super().__init__()
        self.pos_embed = nn.Embedding(num_patches, dim)
        self.patch_proj = nn.Linear(dim, dim)
        self.xformer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=7, batch_first=True)
        self.fc = nn.Linear(num_patches * dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = rearrange(x, 'b (p1 h) (p2 w) -> b (p1 p2) (h w)', p1=4, p2=4)
        x = x + self.pos_embed(torch.arange(x.size(1), device=x.device))
        x = self.patch_proj(x)
        x = self.xformer(x)
        x = x.flatten(1)
        return self.fc(x)


model = Transformer()
checkpoint = torch.load('transformer_mnist_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

random.seed(SEED)
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
"""
nb['cells'][16]['source'] = new16.splitlines(keepends=True)
json.dump(nb, open(NB, 'w'), indent=1)
print('cell 16 updated')
