"""
Fine tuning GPT-2 on the TinyStories dataset.

To train (tested on 8x A100 40GB):
$ python gpt2_tinystories.py

Final checkpoint is saved in ./gpt2-tinystories-final.
Use `gpt2_val_tinystories.ipyn` to evaluate the model.

Rowel Atienza
rowel.atienza@up.edu.ph
2024


References:
1) GPT2 - https://huggingface.co/openai-community/gpt2
2) TinyStories - https://huggingface.co/datasets/roneneldan/TinyStories

"""


import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

import math

# Load dataset
dataset = load_dataset("roneneldan/TinyStories")

# Initialize tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """Tokenize dataset examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# Tokenize datasets
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized_test = dataset["validation"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["validation"].column_names
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

def compute_metrics(eval_pred):
    """Compute perplexity metric."""
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = torch.nn.functional.cross_entropy(
        torch.from_numpy(shift_logits.reshape(-1, shift_logits.shape[-1])),
        torch.from_numpy(shift_labels.reshape(-1))
    )
    
    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")
    
    return {"perplexity": perplexity}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-tinystories",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    #evaluation_strategy="steps",
    #eval_steps=500,
    save_steps=1000,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    #load_best_model_at_end=True,
    #metric_for_best_model="perplexity",
    #greater_is_better=False
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    #eval_dataset=tokenized_test,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save model
trainer.save_model("./gpt2-tinystories-final")