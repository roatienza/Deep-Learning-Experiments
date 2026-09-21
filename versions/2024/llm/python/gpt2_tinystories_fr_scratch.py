"""
Train a small GPT-2 from scratch on the TinyStories dataset.

Unlike ``gpt2_tinystories.py`` (which fine-tunes the pretrained OpenAI GPT-2
weights), this script builds a *smaller* GPT-2 architecture with **random
weights** and trains it from zero.  The goal is to observe how far a tiny
transformer can get on a clean, synthetic corpus of simple children's stories
without any pretraining.

Architecture (GPT-2 "nano" variant):
    - vocab_size : 50,257  (GPT-2 byte-level BPE vocabulary)
    - n_positions: 512     (max context length)
    - n_embd     : 256     (hidden / embedding dimension)
    - n_layer    : 6       (transformer blocks)
    - n_head     : 8       (attention heads per block)
    - parameters : ~11 M   (vs. 124 M for the full GPT-2)

Training setup:
    - Dataset       : roneneldan/TinyStories  (~2.5 M short stories, ~1.2 B tokens)
    - Tokenizer     : GPT-2 Fast tokenizer (byte-level BPE)
    - Batch size    : 2 per device, gradient accumulation x 8  -> effective 16
    - Learning rate : 1e-4 (higher than fine-tuning; random init needs a bigger step)
    - Weight decay  : 0.01
    - Warmup        : 500 steps
    - Epochs        : 5
    - Optimizer     : AdamW (default in HuggingFace Trainer)
    - Scheduler     : linear warmup -> linear decay (default)

Hardware (tested on 8x A100 40GB):
    $ python gpt2_tinystories_fr_scratch.py

The final checkpoint is saved to ``./gpt2-tinystories-fr-scratch-final``.
Use ``gpt2_val_tinystories_fr_scratch.ipynb`` to evaluate and generate text.

Rowel Atienza
rowel.atienza@up.edu.ph
2024

References:
1) GPT-2 paper  - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
2) GPT-2 HF     - https://huggingface.co/openai-community/gpt2
3) TinyStories   - https://huggingface.co/datasets/roneneldan/TinyStories
4) TinyStories   - https://arxiv.org/abs/2305.07759
"""

import math

import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


# ---------------------------------------------------------------------------
# 1. Load the TinyStories dataset
# ---------------------------------------------------------------------------
# TinyStories is a synthetic corpus of ~2.5 M very short children's stories
# (~1.2 B tokens total).  Because the vocabulary is deliberately restricted to
# ~5k unique words, a small model can memorise enough grammar to produce
# coherent, grammatical sentences -- making it ideal for a from-scratch run.
dataset = load_dataset("roneneldan/TinyStories")


# ---------------------------------------------------------------------------
# 2. Tokenizer
# ---------------------------------------------------------------------------
# We reuse the GPT-2 byte-level BPE tokenizer (50,257 tokens).  The pad token
# is set to the EOS token so that padded positions are ignored by the loss.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------------
# 3. Model configuration  (small GPT-2, ~11 M params)
# ---------------------------------------------------------------------------
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,   # 50,257
    n_positions=512,                   # max sequence length
    n_embd=256,                        # hidden dimension
    n_layer=6,                         # number of transformer blocks
    n_head=8,                          # attention heads per block
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Initialise the model with **random** weights (no pretrained checkpoint).
# This is the key difference from gpt2_tinystories.py, which loads the
# pretrained OpenAI GPT-2 weights via from_pretrained("gpt2").
model = GPT2LMHeadModel(config)
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
        token-ID lists of length ``n_positions`` (512).  Sequences shorter
        than 512 are right-padded with the pad (EOS) token; longer ones are
        truncated.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.n_positions,
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

tokenized_test = dataset["validation"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
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
def compute_metrics(eval_pred):
    """Compute perplexity from the model's logits and labels.

    Perplexity = exp(mean cross-entropy loss).  A perplexity of 1 means the
    model is perfectly certain; a perplexity of V (vocab size) means it is
    uniformly random.  Lower is better.

    Args:
        eval_pred: tuple (logits, labels) from the Trainer's evaluation loop.
            logits: numpy array of shape (batch, seq_len, vocab_size)
            labels: numpy array of shape (batch, seq_len)

    Returns:
        dict with a single "perplexity" key.
    """
    logits, labels = eval_pred

    # Shift so that position i predicts position i+1 (causal LM convention).
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross-entropy loss over all non-padding positions.
    loss = torch.nn.functional.cross_entropy(
        torch.from_numpy(shift_logits.reshape(-1, shift_logits.shape[-1])),
        torch.from_numpy(shift_labels.reshape(-1)),
    )

    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")

    return {"perplexity": perplexity}


# ---------------------------------------------------------------------------
# 7. Training arguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./gpt2-tinystories",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 2 * 8 = 16
    evaluation_strategy="steps",
    eval_steps=500,                  # evaluate every 500 steps
    save_steps=1000,                 # save a checkpoint every 1000 steps
    warmup_steps=500,                # linear LR warmup for the first 500 steps
    learning_rate=1e-4,              # higher than fine-tuning (5e-5) because
                                     # we start from random weights
    weight_decay=0.01,               # L2 regularisation on non-bias params
    logging_dir="./logs",
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
trainer.train()


# ---------------------------------------------------------------------------
# 10. Save the final model
# ---------------------------------------------------------------------------
trainer.save_model("./gpt2-tinystories-fr-scratch-final")
print("Model saved to ./gpt2-tinystories-fr-scratch-final")
