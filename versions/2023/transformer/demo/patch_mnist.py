"""Apply the same improvements as the CNN/RNN notebooks to transformer_mnist.ipynb.

Changes:
- Cell 4 (sample visualization): seeded RNG for reproducibility.
- Cell 5 (model markdown): document the learnable positional embedding and
  the patch->d_model projection.
- Cell 6 (model code): add nn.Embedding(16, 49) positional embedding and
  nn.Linear(49, 49) patch projection; keep the same architecture otherwise.
- Cell 17 (summary): update parameter count and architecture row.
"""
import json

NB = 'transformer_mnist.ipynb'

nb = json.load(open(NB))

# ---------------------------------------------------------------- cell 4
old4 = ''.join(nb['cells'][4]['source'])
assert 'np.random.choice' in old4
new4 = old4.replace(
    "indices = np.random.choice(len(train_dataset), size=16, replace=False)",
    "rng = np.random.default_rng(SEED)\n"
    "indices = rng.choice(len(train_dataset), size=16, replace=False)"
)
assert new4 != old4
nb['cells'][4]['source'] = new4.splitlines(keepends=True)

# ---------------------------------------------------------------- cell 5
new5 = """### 3. Build the Transformer model

Architecture:

```
input (1 x 28 x 28)
  -> patchify 4x4 (each patch 7x7) -> (B, 16, 49)
  -> + learnable positional embedding (16, 49)
  -> Linear(49, 49) patch projection
  -> TransformerEncoderLayer(d_model=49, nhead=7, batch_first=True)
  -> flatten (16 x 49 = 784)
  -> Linear(784, 10) -> logits
```

Notes:

* **Patchify** with `einops.rearrange`: `'b (p1 h) (p2 w) -> b (p1 p2) (h w)'`
  with `p1=p2=4` splits the 28x28 image into a 4x4 grid of 7x7 patches,
  producing 16 tokens of 49 pixels each.
* **Positional embedding**: a learnable `nn.Embedding(16, 49)` is added to the
  patch tokens. Self-attention is permutation-invariant, so without a position
  signal the model cannot tell which patch is where; the embedding lets it
  recover the 2D layout from the 16-token sequence.
* **Patch projection**: a `Linear(49, 49)` layer maps each raw 49-pixel patch
  into the model's embedding space before the encoder layer (the ViT-style
  "patch embedding" linear).
* **One encoder layer** with `d_model=49` and `nhead=7` (49 = 7 x 7, so the
  model dimension divides evenly by the head count). Every patch attends to
  every other patch, which gives the model a global view of the digit.
* The **output layer has no activation** -- we return raw logits because
  `nn.CrossEntropyLoss` applies log-softmax internally.
"""
nb['cells'][5]['source'] = new5.splitlines(keepends=True)

# ---------------------------------------------------------------- cell 6
new6 = """import torch
import torch.nn as nn
from einops import rearrange


class Transformer(nn.Module):
    # Patch-based Transformer for MNIST: 4x4 patches -> 1 encoder layer -> head.

    def __init__(self, dim=49, num_classes=10, num_patches=16):
        super().__init__()
        self.pos_embed = nn.Embedding(num_patches, dim)
        self.patch_proj = nn.Linear(dim, dim)
        self.xformer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=7,
            batch_first=True,
        )
        self.fc = nn.Linear(num_patches * dim, num_classes)

    def forward(self, x):
        # (B, 1, 28, 28) -> (B, 28, 28)
        x = x.squeeze(1)
        # (B, 28, 28) -> (B, 16, 49): 4x4 grid of 7x7 patches
        x = rearrange(x, 'b (p1 h) (p2 w) -> b (p1 p2) (h w)', p1=4, p2=4)
        # Add position signal (self-attention is permutation-invariant)
        x = x + self.pos_embed(torch.arange(x.size(1), device=x.device))
        # Project raw pixels into the model's embedding space
        x = self.patch_proj(x)
        x = self.xformer(x)
        x = x.flatten(1)
        return self.fc(x)


model = Transformer()
x = torch.randn(64, 1, 28, 28)
print(model)
print("output shape:", model(x).shape)

num_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {num_params:,}")
"""
nb['cells'][6]['source'] = new6.splitlines(keepends=True)

# ---------------------------------------------------------------- cell 17
old17 = ''.join(nb['cells'][17]['source'])
new17 = old17.replace(
    "| Architecture | Patch Transformer: 4x4 patches (7x7) -> 1 encoder layer (d_model 49, nhead 7) -> Linear(784, 10) |",
    "| Architecture | Patch Transformer: 4x4 patches (7x7) + positional embedding + patch projection -> 1 encoder layer (d_model 49, nhead 7) -> Linear(784, 10) |"
).replace(
    "| Parameters | ~20k |",
    "| Parameters | ~24k |"
)
assert new17 != old17
nb['cells'][17]['source'] = new17.splitlines(keepends=True)

json.dump(nb, open(NB, 'w'), indent=1)
print('patched OK')
