"""Pin the transformer demo notebooks to GPUs 6 and 7.

- transformer_mnist.ipynb: CUDA_VISIBLE_DEVICES='6' (single-GPU notebook)
- transformer_cifar10.ipynb: CUDA_VISIBLE_DEVICES='6,7' + devices=2 (DDP over 2 GPUs)

Idempotent: skips a patch if the marker is already present.
"""
import json

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'

# ---------------------------------------------------------------- MNIST
path = f'{HERE}/transformer_mnist.ipynb'
nb = json.load(open(path))

src = ''.join(nb['cells'][8]['source'])
if "CUDA_VISIBLE_DEVICES" not in src:
    src = src.replace(
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "# Pin to GPU 6 (set before importing torch on a fresh kernel)\n"
        "import os\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = '6'\n\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    )
    nb['cells'][8]['source'] = src.splitlines(keepends=True)

src = ''.join(nb['cells'][12]['source'])
if "weights_only=True" not in src:
    src = src.replace(
        "checkpoint = torch.load('transformer_mnist_best.pt', map_location=device)",
        "checkpoint = torch.load('transformer_mnist_best.pt', map_location=device, weights_only=True)",
    )
    nb['cells'][12]['source'] = src.splitlines(keepends=True)

json.dump(nb, open(path, 'w'), indent=1)
print('patched', path)

# ---------------------------------------------------------------- CIFAR10
path = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(path))

src = ''.join(nb['cells'][1]['source'])
if "CUDA_VISIBLE_DEVICES" not in src:
    src = "# Pin to GPUs 6 and 7 (set before importing torch on a fresh kernel)\n" \
          "import os\n" \
          "os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'\n\n" + src
    nb['cells'][1]['source'] = src.splitlines(keepends=True)

src = ''.join(nb['cells'][13]['source'])
if "default=2" not in src:
    old = "parser.add_argument('--devices', default=1, type=int, metavar='N')"
    assert old in src
    src = src.replace(old, "parser.add_argument('--devices', default=2, type=int, metavar='N')")
    nb['cells'][13]['source'] = src.splitlines(keepends=True)

src = ''.join(nb['cells'][17]['source'])
if "CUDA_VISIBLE_DEVICES" not in src:
    src = src.replace(
        "import os\nimport random",
        "import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '6,7'\nimport random",
    )
    nb['cells'][17]['source'] = src.splitlines(keepends=True)

json.dump(nb, open(path, 'w'), indent=1)
print('patched', path)
