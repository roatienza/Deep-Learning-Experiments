"""Fix stale intro text in cell 0 of transformer_mnist.ipynb: the model now
has a learnable positional embedding and a patch projection."""
import json

NB = 'transformer_mnist.ipynb'
nb = json.load(open(NB))

old = ''.join(nb['cells'][0]['source'])

old_arch = """**Architecture.** 4x4 patches of 7x7 pixels -> 16 tokens of dimension 49 ->
one `TransformerEncoderLayer` (d_model=49, nhead=7) -> flatten (16 x 49) ->
`Linear(784, 10)`. No learned positional embeddings are added; the patch
order (row-major) is the only position signal, which is enough for MNIST."""
new_arch = """**Architecture.** 4x4 patches of 7x7 pixels -> 16 tokens of dimension 49,
plus a learnable positional embedding (self-attention is permutation-
invariant, so the model needs a position signal) -> a linear patch
projection -> one `TransformerEncoderLayer` (d_model=49, nhead=7) ->
flatten (16 x 49) -> `Linear(784, 10)`."""
assert old_arch in old
old = old.replace(old_arch, new_arch)

nb['cells'][0]['source'] = old.splitlines(keepends=True)
json.dump(nb, open(NB, 'w'), indent=1)
print('cell 0 intro fixed')
