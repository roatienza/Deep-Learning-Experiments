"""Patch mlp_mnist.ipynb: add a 1x4 sample-image grid (GT vs prediction)
loaded from the best checkpoint, and update the intro/summary docs.

Run:  python patch_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, 'mlp_mnist.ipynb')

nb = json.load(open(NB_PATH))
cells = nb['cells']


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


# ------------------------------------------------------------------
# 1. Update the intro (cell 0): mention the sample grid
intro = ''.join(cells[0]['source'])
intro = intro.replace(
    "5. Evaluate it: overall accuracy, per-class accuracy, and a confusion matrix.\n"
    "6. Save the best checkpoint and inspect a few predictions.",
    "5. Evaluate it: overall accuracy, per-class accuracy, and a confusion matrix.\n"
    "6. Save the best checkpoint and inspect a few predictions, including a\n"
    "   1x4 grid of sample images with their ground-truth and predicted labels.",
)
cells[0] = md(intro)

# ------------------------------------------------------------------
# 2. Insert the new 1x4 grid section after the confusion-matrix cell
#    (the last code cell of section 7). Find it: the code cell whose
#    source contains 'Confusion matrix (MNIST test set)'.
cm_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'Confusion matrix (MNIST test set)' in ''.join(c['source']):
        cm_idx = i
        break
assert cm_idx is not None, "confusion-matrix cell not found"

new_md = md("""\
### 7.5. Sample images from the checkpoint: ground truth vs prediction

A quick visual check of the trained model. We load the **best checkpoint**
(`mlp_mnist_best.pt`, selected on validation accuracy) and run four random
test images through it. Each panel shows the image with two labels:

* **GT** — the ground-truth digit from the test set.
* **Pred** — the model's prediction (argmax over the 10 logits).

The title is green when the two agree and red when they disagree. On a
well-trained model nearly all panels are green; the occasional red one is
useful for spotting which digit pairs the MLP confuses (typically 4/9,
3/8, or 5/6).
""")

new_code = code("""\
# Load the best checkpoint (already loaded in section 6; reload to be
# self-contained in case this cell is run after a kernel restart).
checkpoint = torch.load('mlp_mnist_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Pick 4 random test images
indices = random.sample(range(len(test_dataset)), 4)
images = [test_dataset[i][0] for i in indices]
labels = [test_dataset[i][1] for i in indices]

with torch.no_grad():
    outputs = model(torch.stack(images).to(device))
    predicted = outputs.argmax(dim=1).cpu().tolist()

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ok = predicted[i] == labels[i]
    ax.set_title(f"GT: {labels[i]}   Pred: {predicted[i]}  {'OK' if ok else 'X'}",
                 color='green' if ok else 'red', fontsize=12)
    ax.axis('off')
plt.suptitle(f"Sample predictions from checkpoint (epoch {checkpoint['epoch']}, "
             f"val acc {checkpoint['val_acc']:.4f})", fontsize=11)
plt.tight_layout()
plt.show()
""")

cells.insert(cm_idx + 1, new_md)
cells.insert(cm_idx + 2, new_code)

# ------------------------------------------------------------------
# 3. Update the summary cell: mention the sample grid
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and c['source'][0].startswith('### Summary'):
        s = ''.join(c['source'])
        s = s.replace(
            "The MLP is a strong baseline: a CNN reaches ~99.2% on the same data, and a\n"
            "Transformer ~99.4%, so the gap to close is small — but the MLP gets you most\n"
            "of the way there with a few hundred thousand parameters and no convolutions.",
            "The MLP is a strong baseline: a CNN reaches ~99.2% on the same data, and a\n"
            "Transformer ~99.4%, so the gap to close is small — but the MLP gets you most\n"
            "of the way there with a few hundred thousand parameters and no convolutions.\n"
            "The 1x4 sample grid in section 7.5 gives a quick visual sanity check of the\n"
            "checkpoint's predictions against the ground-truth labels.",
        )
        cells[i] = md(s)
        break

nb['cells'] = cells
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
print(f"patched {NB_PATH} with {len(cells)} cells")
