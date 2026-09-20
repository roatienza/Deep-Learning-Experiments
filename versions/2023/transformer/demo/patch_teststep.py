"""Fix test_step for DDP: accumulate into the metric (auto-synchronized)
and compute+log in test_epoch_end, instead of stacking per-batch tensors."""
import json

path = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo/transformer_cifar10.ipynb'
nb = json.load(open(path))
src = ''.join(nb['cells'][13]['source'])

old = '''    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return {"y_hat": y_hat, "test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc*100., on_epoch=True, prog_bar=True)'''
new = '''    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # Accumulate into the metric; it is synchronized across DDP ranks
        # automatically, so test_epoch_end only needs to compute + log.
        self.accuracy(y_hat, y)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.accuracy.compute() * 100., on_epoch=True, prog_bar=True)
        self.accuracy.reset()'''
assert old in src, 'test_step block not found'
src = src.replace(old, new)
nb['cells'][13]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(path, 'w'), indent=1)
print('patched test_step')
