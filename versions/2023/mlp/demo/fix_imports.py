"""Fix imports in the 1x4 grid cell (cell 17) of mlp_mnist.ipynb."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, 'mlp_mnist.ipynb')

nb = json.load(open(NB_PATH))
src = ''.join(nb['cells'][17]['source'])
if 'import random' not in src:
    src = 'import random\nimport torch\nimport matplotlib.pyplot as plt\n\n' + src
    nb['cells'][17]['source'] = src.splitlines(keepends=True)
    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')
    print('fixed imports in cell 17')
else:
    print('imports already present')
