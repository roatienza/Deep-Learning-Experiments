"""Fix the DDP + torchmetrics sync deadlock in LitTransformer.
Replace on_validation_epoch_end with a version that uses the already-synced
val_acc from callback_metrics instead of calling self.accuracy.compute().
"""
import json

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

src = ''.join(nb['cells'][13]['source'])

# Find and replace the on_validation_epoch_end method
import re
pattern = r'    def on_validation_epoch_end\(self\):\n        self\.log\("val_acc_epoch", self\.accuracy\.compute\(\)\*100\., on_epoch=True, prog_bar=True\)'
replacement = '''    def on_validation_epoch_end(self):
        # Use the already-synced val_acc metric to avoid a DDP deadlock.
        # self.accuracy.compute() triggers a cross-rank all-reduce that
        # deadlocks when called inside a Lightning hook under DDP.
        val_acc = self.trainer.callback_metrics.get("val_acc", torch.tensor(0.0))
        self.log("val_acc_epoch", val_acc * 100., on_epoch=True, prog_bar=True)'''

new_src, count = re.subn(pattern, replacement, src)
assert count == 1, f"Expected 1 replacement, got {count}"
nb['cells'][13]['source'] = new_src.splitlines(keepends=True)
json.dump(nb, open(p, 'w'), indent=1)
print('FIXED DDP deadlock (regex)')
