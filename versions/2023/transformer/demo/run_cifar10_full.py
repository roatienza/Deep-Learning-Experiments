"""Run the full transformer_cifar10.ipynb notebook headlessly on GPU 6 and 7.
Executes all cells in order, captures the training output, and reports
the test accuracy from the final cell.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Use GPU 6 and 7
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

nb = json.load(open('transformer_cifar10.ipynb'))

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
