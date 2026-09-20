"""Run the CIFAR10 notebook headlessly on GPUs 6 and 7 (DDP, 2 ranks).

Patches the notebook in memory before execution:
- NCCL_DEBUG=WARN in the main cell (diagnosable hangs)
- num_workers=2 (CPU is shared with other jobs; many workers x 2 ranks
  oversubscribed it and stalled the DataLoader)
- max_epochs=3 (verification run; the notebook default stays 10)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Use GPU 6 and 7
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['NCCL_DEBUG'] = 'WARN'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

nb = json.load(open('transformer_cifar10.ipynb'))

# In-memory patches for this verification run
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'if __name__ == "__main__":' in src:
        src = src.replace(
            'os.environ.setdefault("NCCL_DEBUG", "WARN")',
            'os.environ["NCCL_DEBUG"] = "WARN"')
        src = src.replace(
            'num_workers=args.num_workers)',
            'num_workers=2)')
        src = src.replace(
            'args = get_args()',
            'args = get_args()\n    args.max_epochs = 3  # verification run')
        cell['source'] = src.splitlines(keepends=True)

# Shared globals so cells can see each other's definitions
g = {'__name__': '__main__'}

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if not src.strip():
        continue
    print(f'--- cell {i} ---', flush=True)
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), g)
    except Exception as e:
        print(f'ERROR in cell {i}: {type(e).__name__}: {e}', file=sys.stderr)
        raise

print('ALL CELLS OK', flush=True)
