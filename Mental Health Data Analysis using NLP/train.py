import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import evaluate  # Fix: Correct import for metrics
import nlpaug.augmenter.word as naw
from torch.utils.tensorboard import SummaryWriter
import os

# Load Dataset
dataset = load_dataset("emotion")  # Ensure you have the dataset installed

# Load Pretrained Model & Tokenizer
MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)

# Data Augmentation (Synonym replacement)
def augment_text(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    return aug.augment(text)

def preprocess_data(examples):
    texts = examples["text"]
    labels = examples["label"]
    augmented_texts = [augment_text(text)[0] for text in texts]  # Ensure single string output
    encodings = tokenizer(augmented_texts, padding="max_length", truncation=True)
    encodings["labels"] = labels  # Assign labels correctly
    return encodings

dataset = dataset.map(preprocess_data, batched=True)

# Class Weights for Imbalanced Data
class_counts = np.bincount(dataset["train"]["label"])
weights = 1.0 / class_counts
class_weights = torch.tensor(weights, dtype=torch.float32)

# Load Evaluation Metric (Fixed `load_metric` → `evaluate.load`)
accuracy = evaluate.load("accuracy")

# Define Compute Metrics Function
def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=p.label_ids)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Enable mixed precision for faster training
    report_to="tensorboard",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Fixed usage of metrics
)

# Train Model
trainer.train()

# Save Best Model
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")
print("Training Complete. Best model saved!")
