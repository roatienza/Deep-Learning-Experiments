"""Fix the DDP + torchmetrics sync deadlock in LitTransformer.
Replace on_validation_epoch_end with a no-op (val_acc_epoch is already logged
by Lightning from the val_acc metric in validation_step).
"""
import json
import re

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

src = ''.join(nb['cells'][13]['source'])

# Find the on_validation_epoch_end method and replace it with a no-op
pattern = r'    def on_validation_epoch_end\(self\):\n(?:        .*\n)*?        self\.log\("val_acc_epoch".*\n'
replacement = '''    def on_validation_epoch_end(self):
        # No-op: val_acc_epoch is already logged by Lightning from the
        # val_acc metric in validation_step. Calling self.accuracy.compute()
        # here would trigger a cross-rank all-reduce that deadlocks under DDP.
        pass
'''

new_src, count = re.subn(pattern, replacement, src)
assert count == 1, f"Expected 1 replacement, got {count}"
nb['cells'][13]['source'] = new_src.splitlines(keepends=True)
json.dump(nb, open(p, 'w'), indent=1)
print('FIXED DDP deadlock (no-op)')
