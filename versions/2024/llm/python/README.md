# GPT-2 on TinyStories — Training & Validation Results

Two experiments train a small GPT-2 on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) corpus
(~2.5 M short children's stories, ~1.2 B tokens):

| File | What it does |
| :--- | :--- |
| [`gpt2_tinystories_fr_scratch.py`](gpt2_tinystories_fr_scratch.py) | Trains a **~11 M-parameter** GPT-2 ("nano") from **random weights** |
| [`gpt2_tinystories.py`](gpt2_tinystories.py) | **Fine-tunes** the pretrained OpenAI GPT-2 weights |
| [`gpt2_val_tinystories_fr_scratch.ipynb`](gpt2_val_tinystories_fr_scratch.ipynb) | Loads the trained checkpoint, reports perplexity, and generates sample stories |
| [`gpt2_val_tinystories.ipynb`](gpt2_val_tinystories.ipynb) | Same, for the fine-tuned model |

Both runs were done on **8× A100 40 GB**.

## Architecture (both scripts)

| | |
| :--- | :--- |
| Vocab size | 50,257 (GPT-2 byte-level BPE) |
| Context length | 512 |
| Hidden dim (`n_embd`) | 256 |
| Layers (`n_layer`) | 6 |
| Heads (`n_head`) | 8 |
| Parameters | ~11 M (vs. 124 M for full GPT-2) |

## Training setup

| | From scratch | Fine-tune |
| :--- | :---: | :---: |
| Init | random weights | OpenAI GPT-2 |
| Epochs | 5 | 3 (script default) |
| Per-device batch | 2 | 2 |
| Gradient accumulation | ×8 (effective 16) | ×8 (effective 16) |
| Learning rate | 1e-4 | 5e-5 |
| Warmup steps | 500 | 500 |
| Weight decay | 0.01 | 0.01 |
| Optimizer | AdamW | AdamW |
| Scheduler | linear warmup → linear decay | linear warmup → linear decay |
| Eval cadence | every 500 steps | — |
| Eval set | 1,000 validation samples (OOM guard) | — |

The from-scratch run logged **82,805 optimizer steps** over 5 epochs
(~16,561 steps/epoch) and ~1.54 × 10¹⁷ FLOPs.

## Results — from-scratch run

The numbers below come from the run's `trainer_state.json`
(`gpt2-tinystories/checkpoint-82805`). Validation perplexity is
`exp(eval_loss)`, computed on the 1,000-sample validation subset every 500
steps.

### Validation (held-out)

| Epoch | Step | Eval loss | Perplexity |
| :---: | ---: | ---: | ---: |
| 0.03 | 500 | 4.3507 | 79.80 |
| 1 | 17,000 | 1.6964 | 5.48 |
| 2 | 33,500 | 1.5831 | 4.89 |
| 3 | 50,000 | 1.5382 | 4.67 |
| 4 | 66,500 | 1.5128 | 4.55 |
| **5 (final)** | **82,805** | **1.5001** | **4.49** |

### Training loss

| Epoch | Avg. train loss |
| :---: | ---: |
| 0 | 2.7800 |
| 1 | 1.8431 |
| 2 | 1.7014 |
| 3 | 1.6474 |
| 4 | 1.6185 |
| 5 | 1.6061 |

Training loss fell from **10.19** at step 100 to **1.61** at step 82,800.

### Takeaways

- **Perplexity dropped from ~80 to ~4.5** over the run — the model went from
  near-random to producing coherent, on-topic TinyStories.
- **Learning was front-loaded:** the bulk of the gain happened in the first
  epoch (perplexity 79.8 → 5.5). Epochs 2–5 each shrank the gap by only ~0.2–0.3,
  so the model had largely converged by the end of epoch 1.
- **No overfitting:** validation perplexity kept falling (4.89 → 4.55 → 4.49)
  across the last three epochs while training loss kept dropping, so the model
  was still generalizing, not memorizing.
- **Train–eval gap ≈ 0.10** (1.61 vs. 1.50) at the end — a healthy, small gap
  for a language model on a clean, repetitive synthetic corpus.

## Fine-tuned run

`gpt2_tinystories.py` fine-tunes the full pretrained GPT-2 (124 M params) for
3 epochs and saves the result to `gpt2-tinystories-final`. It uses a higher
perplexity budget (full GPT-2, no eval set wired up in the script), so it is
not directly comparable to the 11 M from-scratch numbers above. Use
`gpt2_val_tinystories.ipynb` to load `gpt2-tinystories-final` and generate
stories.

## Reproducing

```bash
# from scratch (~11 M params)
python gpt2_tinystories_fr_scratch.py
# final checkpoint -> ./gpt2-tinystories-fr-scratch-final
# evaluate / generate
# open gpt2_val_tinystories_fr_scratch.ipynb

# fine-tune pretrained GPT-2
python gpt2_tinystories.py
# final checkpoint -> ./gpt2-tinystories-final
# evaluate / generate
# open gpt2_val_tinystories.ipynb
```
