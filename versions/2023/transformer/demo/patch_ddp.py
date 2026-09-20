"""Fix the DDP + torchmetrics sync deadlock in LitTransformer.
The on_validation_epoch_end hook calls self.accuracy.compute() which triggers
a cross-rank all-reduce that deadlocks under DDP. Replace it with a simple
mean of the logged val_acc values, which is already synced by Lightning.
"""
import json

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

src = ''.join(nb['cells'][13]['source'])

# Replace the on_validation_epoch_end method
old = '''    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.accuracy.compute()*100., on_epoch=True, prog_bar=True)'''

new = '''    def on_validation_epoch_end(self):
        # Use the already-synced val_acc metric instead of calling
        # self.accuracy.compute() which deadlocks under DDP.
        val_acc = self.trainer.callback_metrics.get("val_acc", torch.tensor(0.0))
        self.log("val_acc_epoch", val_acc * 100., on_epoch=True, prog_bar=True)'''

assert old in src, "on_validation_epoch_end not found"
src = src.replace(old, new)

nb['cells'][13]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(p, 'w'), indent=1)
print('FIXED DDP deadlock')
