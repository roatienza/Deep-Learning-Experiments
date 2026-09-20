"""Fix the DDP + torchmetrics sync deadlock in LitTransformer.
Replace on_validation_epoch_end with a version that logs val_acc_epoch
from the already-synced val_acc metric (no cross-rank all-reduce).
"""
import json
import re

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

src = ''.join(nb['cells'][13]['source'])

# Find the on_validation_epoch_end method and replace it
pattern = r'    def on_validation_epoch_end\(self\):\n(?:        .*\n)*?        pass\n'
replacement = '''    def on_validation_epoch_end(self):
        # Log val_acc_epoch from the already-synced val_acc metric.
        # This avoids calling self.accuracy.compute() which triggers a
        # cross-rank all-reduce that deadlocks under DDP.
        val_acc = self.trainer.callback_metrics.get("val_acc", torch.tensor(0.0))
        self.log("val_acc_epoch", val_acc * 100., on_epoch=True, prog_bar=True)
'''

new_src, count = re.subn(pattern, replacement, src)
assert count == 1, f"Expected 1 replacement, got {count}"
nb['cells'][13]['source'] = new_src.splitlines(keepends=True)
json.dump(nb, open(p, 'w'), indent=1)
print('FIXED DDP deadlock (log from callback_metrics)')
