"""Run the improved mamba2 notebook headlessly and report results."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import matplotlib
matplotlib.use('Agg')

# Skip the `!pip install` cell (notebook-only magic)
def _skip_pip(src):
    return src.startswith('!pip')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

nb = json.load(open('mamba2_mnist.ipynb'))

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if _skip_pip(src):
        print(f'--- cell {i} (skipped: pip install) ---', flush=True)
        continue
    print(f'--- cell {i} ---', flush=True)
    g = {'__name__': '__main__'} if i == 0 else globals()
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), g)
    except Exception as e:
        print(f'ERROR in cell {i}: {type(e).__name__}: {e}', file=sys.stderr)
        raise

print('ALL CELLS OK')
