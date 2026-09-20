"""Fix the double-setup bug in LitCifar10: the datamodule.setup() call in cell 15
consumes the random_split, so the Trainer's internal setup() call fails because
train_set is already a Subset. Make setup() idempotent by guarding with a flag.
"""
import json

HERE = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo'
p = f'{HERE}/transformer_cifar10.ipynb'
nb = json.load(open(p))

src = ''.join(nb['cells'][13]['source'])

# Add a _split_done flag in __init__
src = src.replace(
    '        self.seed = seed',
    '        self.seed = seed\n        self._split_done = False'
)

# Guard the split in setup()
src = src.replace(
    '''    def setup(self, stage=None):
        # Split the official 50k train set into train / validation once.
        if stage in (None, 'fit'):
            self.train_set, self.val_set = torch.utils.data.random_split(
                self.train_set, [45000, 5000],
                generator=torch.Generator().manual_seed(self.seed))''',
    '''    def setup(self, stage=None):
        # Split the official 50k train set into train / validation once.
        # Guard against being called twice (once by the user, once by the Trainer).
        if stage in (None, 'fit') and not self._split_done:
            self.train_set, self.val_set = torch.utils.data.random_split(
                self.train_set, [45000, 5000],
                generator=torch.Generator().manual_seed(self.seed))
            self._split_done = True'''
)

nb['cells'][13]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(p, 'w'), indent=1)
print('FIXED double-setup bug')
