"""Log val_acc in validation_step so ModelCheckpoint can monitor it under DDP."""
import json

path = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo/transformer_cifar10.ipynb'
nb = json.load(open(path))
src = ''.join(nb['cells'][13]['source'])

old = '''        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return self.test_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        # No-op: val_acc is already logged by Lightning in validation_step.
        # Calling self.accuracy.compute() or self.log() here triggers a
        # cross-rank all-reduce that deadlocks under DDP.
        pass'''
new = '''        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Log val_acc here (not in on_validation_epoch_end): the accuracy
        # metric is synchronized automatically across DDP ranks, so
        # ModelCheckpoint can monitor it without a manual all-reduce.
        self.log("val_acc", self.accuracy(y_hat, y) * 100.,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.accuracy.reset()'''
assert old in src, 'validation_step block not found'
src = src.replace(old, new)
nb['cells'][13]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(path, 'w'), indent=1)
print('patched validation_step')
