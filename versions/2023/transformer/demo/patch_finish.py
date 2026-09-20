"""Finalize transformer_cifar10.ipynb:
1. cell 1: add SEED
2. cell 13: weight decay in optimizer
3. cell 15: devices -> 2
4. cell 14: add 'in this run' note
"""
import json

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

def set_src(i, text):
    nb['cells'][i]['source'] = text.splitlines(keepends=True)

# 1. cell 1: add SEED after imports
src = ''.join(nb['cells'][1]['source'])
assert 'SEED' not in src
src = src.replace(
    'from torchvision.datasets.cifar import CIFAR10',
    'from torchvision.datasets.cifar import CIFAR10\n\n# Reproducibility\nSEED = 42\ntorch.manual_seed(SEED)\ntorch.cuda.manual_seed_all(SEED)')
set_src(1, src)

# 3. cell 15: devices -> 2
src = ''.join(nb['cells'][15]['source'])
assert 'devices=args.devices' in src
src = src.replace('devices=args.devices', 'devices=2')
set_src(15, src)

# 4. cell 14: note about this run
src = ''.join(nb['cells'][14]['source'])
src = src.rstrip() + (
    '\n\n**In this run** we use depth 12, 4 heads, embed dim 64, 4x4 patches '
    '(sequence length 64), Adam with weight decay 1e-4 and cosine LR annealing, '
    '10 epochs, batch 64, on 2 GPUs with 16-bit AMP. The best checkpoint is '
    'selected on validation accuracy (45k train / 5k val split) and evaluated '
    'once on the 10k test set.'
)
set_src(14, src)

json.dump(nb, open(p, 'w'), indent=1)
print('PATCH OK')
