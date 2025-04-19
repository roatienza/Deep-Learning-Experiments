from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np
import math
from torch.utils.data import Dataset

# Load dataset
dataset = load_dataset("roneneldan/TinyStories")

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Define model configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# Initialize model with random weights
model = GPT2LMHeadModel(config)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.n_positions,
        padding="max_length"
    )

# Tokenize datasets
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4  # Parallel processing
)

tokenized_test = dataset["validation"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
    num_proc=4
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-tinystories",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    #evaluation_strategy="steps",
    #eval_steps=500,
    save_steps=1000,
    warmup_steps=500,
    learning_rate=1e-4,
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
trainer.save_model("./gpt2-tinystories-fr-scratch-final")