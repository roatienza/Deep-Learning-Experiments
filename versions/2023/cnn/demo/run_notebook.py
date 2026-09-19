"""Run the improved CNN notebook headlessly and report results."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

nb = json.load(open('cnn_mnist.ipynb'))

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    print(f'--- cell {i} ---', flush=True)
    g = {'__name__': '__main__'} if i == 0 else globals()
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), g)
    except Exception as e:
        print(f'ERROR in cell {i}: {type(e).__name__}: {e}', file=sys.stderr)
        raise

print('ALL CELLS OK')
