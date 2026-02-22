import torch
from datasets import concatenate_datasets
from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification,
    Trainer,
    TrainingArguments,
)
from src.dataset import build_dataset

LABEL2ID = {
    "drawings": 0,
    "hentai": 1,
    "normal": 2,
    "pornographic": 3,
    "suggestive": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

#loading all the datasets sihbwihebchiwebwehbhi
datasets = [
    build_dataset("data/drawings/urls_drawings.txt", LABEL2ID["drawings"]),
    build_dataset("data/hentai/urls_hentai.txt", LABEL2ID["hentai"]),
    build_dataset("data/normal/urls_neutral.txt", LABEL2ID["normal"]),
    build_dataset("data/pornographic/urls_pornographic.txt", LABEL2ID["pornographic"]),
    build_dataset("data/suggestive/urls_suggestive.txt", LABEL2ID["suggestive"]),
]

dataset = concatenate_datasets(datasets).shuffle(seed=42)
splits = dataset.train_test_split(test_size=0.1)

train_ds = splits["train"]
eval_ds = splits["test"]
model_name = "google/siglip-base-patch16-256"
processor = AutoImageProcessor.from_pretrained(model_name)

def preprocess(example):
    processed = processor(
        images=example["image"],
        return_tensors="pt"
    )
    processed["labels"] = example["label"]
    return processed

train_ds = train_ds.with_transform(preprocess)
eval_ds = eval_ds.with_transform(preprocess)


model = SiglipForImageClassification.from_pretrained(
    model_name,
    num_labels=5,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

model.to(device)

training_args = TrainingArguments(
    output_dir="models/siglip_nsfw",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()
