"""Improve transformer_cifar10.ipynb (cell 13 holds LitTransformer + LitCifar10
+ get_args; cell 15 is the __main__ training block).

Changes:
- LitTransformer: weight decay in Adam; log train/val/test loss and acc.
- LitCifar10: 45k/5k train/val split, real val_dataloader (was an alias of
  test_dataloader), seeded split, pin_memory.
- __main__: seed, auto accelerator/precision, ModelCheckpoint on val acc,
  fit, then load best checkpoint and test.
"""
import json

NB = 'transformer_cifar10.ipynb'
nb = json.load(open(NB))

# ------------------------------------------------------------- cell 13
old13 = ''.join(nb['cells'][13]['source'])

old_opt = """    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # this decays the learning rate to 0 after max_epochs using cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]"""
new_opt = """    def configure_optimizers(self):
        # weight decay 1e-4 = L2 regularization
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        # this decays the learning rate to 0 after max_epochs using cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]"""
assert old_opt in old13
old13 = old13.replace(old_opt, new_opt)

old_ts = """    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss"""
new_ts = """    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss"""
assert old_ts in old13
old13 = old13.replace(old_ts, new_ts)

old_vs = """    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)"""
new_vs = """    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return self.test_step(batch, batch_idx)"""
assert old_vs in old13
old13 = old13.replace(old_vs, new_vs)

old_dm = """# a lightning data module for cifar 10 dataset
class LitCifar10(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=32, patch_num=4, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.patch_num = patch_num
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_set = CIFAR10(root='~/data', train=True,
                                 download=True, transform=torchvision.transforms.ToTensor())
        self.test_set = CIFAR10(root='~/data', train=False,
                                download=True, transform=torchvision.transforms.ToTensor())

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.LongTensor(y)
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=self.patch_num, p2=self.patch_num)
        return x, y

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, 
                                        shuffle=True, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, 
                                        shuffle=False, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers)

    def val_dataloader(self):
        return self.test_dataloader()"""

new_dm = """# a lightning data module for cifar 10 dataset
class LitCifar10(LightningDataModule):
    # The 50k official training set is split into 45k train / 5k validation.
    # The validation set is used only to monitor generalization and to select
    # the best checkpoint; the 10k test set is touched once, at the end.
    def __init__(self, batch_size=32, num_workers=32, patch_num=4, seed=42, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.patch_num = patch_size = patch_num
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        self.train_set = CIFAR10(root='~/data', train=True,
                                 download=True, transform=torchvision.transforms.ToTensor())
        self.test_set = CIFAR10(root='~/data', train=False,
                                download=True, transform=torchvision.transforms.ToTensor())

    def setup(self, stage=None):
        # Split the official 50k train set into train / validation once.
        if stage in (None, 'fit'):
            self.train_set, self.val_set = torch.utils.data.random_split(
                self.train_set, [45000, 5000],
                generator=torch.Generator().manual_seed(self.seed))

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.LongTensor(y)
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=self.patch_num, p2=self.patch_num)
        return x, y

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                        shuffle=True, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                        shuffle=False, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                        shuffle=False, collate_fn=self.collate_fn,
                                        num_workers=self.num_workers, pin_memory=True)"""

assert old_dm in old13, 'datamodule block not found'
old13 = old13.replace(old_dm, new_dm)

nb['cells'][13]['source'] = old13.splitlines(keepends=True)

# ------------------------------------------------------------- cell 15 main
new15 = """if __name__ == "__main__":
    args = get_args()

    # Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    datamodule = LitCifar10(batch_size=args.batch_size,
                            patch_num=args.patch_num,
                            num_workers=args.num_workers * args.devices)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    sample_data = next(iter(datamodule.train_dataloader()))
    data = sample_data[0][0]
    print(data.shape)

    patch_dim = data.shape[-1]
    seqlen = data.shape[-2]
    print("Embed dim:", args.embed_dim)
    print("Patch size:", 32 // args.patch_num)
    print("Sequence length:", seqlen)

    model = LitTransformer(num_classes=10, lr=args.lr, epochs=args.max_epochs,
                           depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
                           patch_dim=patch_dim, seqlen=seqlen,)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} million")

    # Use the GPU when available, otherwise CPU.
    accelerator = args.accelerator if torch.cuda.is_available() else 'cpu'
    precision = 16 if accelerator == 'gpu' else 32
    print(f"Running on: {accelerator} (precision {precision})")

    # Keep the checkpoint with the best validation accuracy.
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc_epoch", mode="max",
        filename="transformer_cifar10_best", save_top_k=1)

    trainer = Trainer(accelerator=accelerator, devices=args.devices,
                      max_epochs=args.max_epochs, precision=precision,
                      callbacks=[checkpoint_cb],
                      default_root_dir='./lightning_logs')
    trainer.fit(model, datamodule=datamodule)

    # Evaluate the best checkpoint on the held-out test set.
    best_path = checkpoint_cb.best_model_path
    print(f"Best checkpoint: {best_path} (val acc {checkpoint_cb.best_model_score:.4f})")
    best_model = LitTransformer.load_from_checkpoint(
        best_path, num_classes=10, lr=args.lr, epochs=args.max_epochs,
        depth=args.depth, embed_dim=args.embed_dim, head=args.num_heads,
        patch_dim=patch_dim, seqlen=seqlen)
    trainer.test(best_model, datamodule=datamodule, verbose=True)
"""
nb['cells'][15]['source'] = new15.splitlines(keepends=True)

json.dump(nb, open(NB, 'w'), indent=1)
print('cifar10 notebook patched OK')
