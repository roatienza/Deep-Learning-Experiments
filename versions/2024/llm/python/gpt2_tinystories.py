"""
Fine-tune the pretrained OpenAI GPT-2 (124 M parameters) on the TinyStories
dataset.

Unlike ``gpt2_tinystories_fr_scratch.py`` (which builds a small GPT-2 with
random weights and trains it from zero), this script loads the pretrained
OpenAI GPT-2 weights and adapts them to the TinyStories corpus.  The
pretrained model already knows English grammar and vocabulary; fine-tuning
teaches it the *style* of TinyStories -- very simple, short children's
stories with a restricted vocabulary.

Dataset:
    roneneldan/TinyStories -- ~2.5 M very short children's stories
    (~1.2 B tokens).  The vocabulary is deliberately restricted to ~5k
    unique words, so a pretrained model adapts quickly.

Training setup (tested on 8x A100 40GB, but works on a single GPU):
    - Init          : pretrained OpenAI GPT-2 (openai-community/gpt2)
    - Epochs        : 3
    - Batch size    : 2 per device, gradient accumulation x 8
                      -> effective batch size 16
    - Learning rate : 5e-5 (lower than from-scratch; the weights are already
                      sensible and a big step would destroy them)
    - Warmup        : 500 steps, then linear decay
    - Weight decay  : 0.01
    - Eval          : every 500 steps on a 1,000-sample validation subset,
                      reporting token-weighted perplexity

Outputs:
    - Checkpoints   : ./gpt2-tinystories-ft/checkpoint-*
    - Final model   : ./gpt2-tinystories-ft-final
    - Validation    : gpt2_val_tinystories.ipynb (loads the final model,
                      reports perplexity, generates sample stories)

Rowel Atienza
rowel.atienza@up.edu.ph
2024

References:
1) GPT2 - https://huggingface.co/openai-community/gpt2
2) TinyStories - https://huggingface.co/datasets/roneneldan/TinyStories
"""

import math
import os

import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# 1. Load the TinyStories dataset
# ---------------------------------------------------------------------------
dataset = load_dataset("roneneldan/TinyStories")


# ---------------------------------------------------------------------------
# 2. Tokenizer
# ---------------------------------------------------------------------------
# Reuse the GPT-2 byte-level BPE tokenizer (50,257 tokens).  The pad token is
# set to the EOS token so that padded positions are ignored by the loss.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------------
# 3. Model  (pretrained OpenAI GPT-2, 124 M params)
# ---------------------------------------------------------------------------
# This is the key difference from gpt2_tinystories_fr_scratch.py: we load the
# pretrained weights instead of initialising a small model from scratch.
model = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------------
# 4. Tokenisation
# ---------------------------------------------------------------------------
def tokenize_function(examples):
    """Tokenise a batch of raw story strings into fixed-length token IDs.

    Args:
        examples: dict with a "text" key holding a list of raw strings
                  (one per story).

    Returns:
        dict with "input_ids" and "attention_mask" keys, each a list of
        token-ID lists of length 512.  Sequences shorter than 512 are
        right-padded with the pad (EOS) token; longer ones are truncated.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )


# Tokenise the train and validation splits in parallel (4 worker processes).
# ``remove_columns`` drops the original "text" column to save memory.
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4,
)

# Limit the eval set to 1,000 samples.  The Trainer accumulates ALL logits in
# RAM before calling compute_metrics; with 22k samples x 512 tokens x 50k
# vocab that is tens of GB per rank and OOMs.  1,000 samples is enough for a
# stable perplexity estimate.
eval_subset = dataset["validation"].select(range(1000))
tokenized_test = eval_subset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_subset.column_names,
    num_proc=4,
)


# ---------------------------------------------------------------------------
# 5. Data collator
# ---------------------------------------------------------------------------
# For causal (left-to-right) language modelling we need:
#   - input_ids : the token sequence
#   - labels    : the *same* sequence, shifted right by one position
#                 (position i predicts position i+1)
# DataCollatorForLanguageModeling with mlm=False does exactly this and also
# builds the attention mask so that padding tokens do not contribute to loss.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM, not masked LM
)


# ---------------------------------------------------------------------------
# 6. Evaluation metric  (perplexity)
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred, compute_result=False):
    """Compute perplexity from the model's logits and labels.

    Perplexity = exp(mean cross-entropy loss).  A perplexity of 1 means the
    model is perfectly certain; a perplexity of V (vocab size) means it is
    uniformly random.  Lower is better.

    With ``batch_eval_metrics=True`` the Trainer calls this function once per
    batch.  ``compute_result`` is False for every batch except the last, where
    it is True.  We accumulate the per-batch losses and return the aggregate
    perplexity only on the final call.

    Args:
        eval_pred: EvalPrediction with ``predictions`` (logits) and
            ``label_ids`` (labels).  Tensors with ``batch_eval_metrics=True``,
            numpy arrays otherwise.
        compute_result: True on the last eval batch, False otherwise.

    Returns:
        dict with a single "perplexity" key (only on the last batch).
    """
    # Accumulate per-batch loss x count so we can compute the weighted mean.
    if not hasattr(compute_metrics, "_loss_sum"):
        compute_metrics._loss_sum = 0.0
        compute_metrics._token_count = 0

    logits, labels = eval_pred

    # With batch_eval_metrics=True the Trainer passes raw Tensors;
    # without it, numpy arrays.  Normalise to Tensors on CPU.
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels)
    logits = logits.cpu()
    labels = labels.cpu()

    # Shift so that position i predicts position i+1 (causal LM convention).
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Count non-padding tokens (labels != -100) for weighted averaging.
    n_tokens = int((shift_labels != -100).sum())

    # Compute cross-entropy loss over all non-padding positions.
    # (padding positions are already masked to -100 in the labels by the
    #  DataCollatorForLanguageModeling, so they are excluded automatically)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )

    compute_metrics._loss_sum += loss.item() * n_tokens
    compute_metrics._token_count += n_tokens

    if not compute_result:
        return {}

    # Final batch: return the aggregate perplexity and reset accumulators.
    avg_loss = compute_metrics._loss_sum / max(compute_metrics._token_count, 1)
    compute_metrics._loss_sum = 0.0
    compute_metrics._token_count = 0

    # Clamp to avoid overflow: exp(709) is the largest finite float64 value.
    perplexity = math.exp(min(avg_loss, 709.0))

    return {"perplexity": perplexity}


# ---------------------------------------------------------------------------
# 7. Training arguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    # NOTE: separate directory from gpt2_tinystories_fr_scratch.py, whose
    # checkpoints live in ./gpt2-tinystories.
    output_dir="./gpt2-tinystories-ft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 2 * 8 = 16
    eval_strategy="steps",
    eval_steps=500,                  # evaluate every 500 steps
    # batch_eval_metrics=True: compute perplexity per-batch and free logits
    # immediately.  Without this, the Trainer accumulates ALL eval logits on
    # GPU and OOMs.
    batch_eval_metrics=True,
    save_steps=1000,                 # save a checkpoint every 1000 steps
    # Disk is tight: keep only the 3 most recent checkpoints so the run
    # cannot die on a full filesystem mid-map.
    save_total_limit=3,
    warmup_steps=500,                # linear LR warmup for the first 500 steps
    learning_rate=5e-5,              # lower than from-scratch (1e-4) because
                                     # we start from pretrained weights
    weight_decay=0.01,               # L2 regularisation on non-bias params
    logging_steps=100,               # log loss / LR every 100 steps
    load_best_model_at_end=True,     # reload the best checkpoint at the end
    metric_for_best_model="perplexity",
    greater_is_better=False,         # lower perplexity is better
)


# ---------------------------------------------------------------------------
# 8. Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ---------------------------------------------------------------------------
# 9. Train
# ---------------------------------------------------------------------------
# Resume from the last saved checkpoint if present (e.g. after a crash/OOM),
# otherwise start from the pretrained weights.
_ckpt_dir = os.path.join(training_args.output_dir, "checkpoint-16000")
trainer.train(resume_from_checkpoint=_ckpt_dir if os.path.isdir(_ckpt_dir) else None)


# ---------------------------------------------------------------------------
# 10. Save the final model
# ---------------------------------------------------------------------------
trainer.save_model("./gpt2-tinystories-ft-final")
print("Model saved to ./gpt2-tinystories-ft-final")
