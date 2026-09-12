import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------
df = pd.read_csv("data/train.csv")

# Ensure correct column names
df = df[["text", "label"]]
df = df.dropna()

# Train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ---------------------------
# 2️⃣ Load Tokenizer
# ---------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["text", "__index_level_0__"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ---------------------------
# 3️⃣ Load BERT Model
# ---------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# ---------------------------
# 4️⃣ Define Metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# ---------------------------
# 5️⃣ Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# ---------------------------
# 6️⃣ Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ---------------------------
# 7️⃣ Train Model
# ---------------------------
trainer.train()

# ---------------------------
# 8️⃣ Final Evaluation
# ---------------------------
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\nFinal Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# ---------------------------
# 9️⃣ Save Model
# ---------------------------
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("\nBERT model trained and saved successfully.")
