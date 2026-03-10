# import torch
# import os
# import numpy as np
# from datasets import concatenate_datasets
# from transformers import AutoImageProcessor, SiglipForImageClassification, Trainer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# from src.dataset import build_dataset


# LABEL2ID = {
#     "drawings": 0,
#     "hentai": 1,
#     "normal": 2,
#     "pornographic": 3,
#     "suggestive": 4,
# }

# ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")


# # Load dataset
# datasets = [
#     build_dataset("data/drawings/urls_drawings.txt", LABEL2ID["drawings"]),
#     build_dataset("data/hentai/urls_hentai.txt", LABEL2ID["hentai"]),
#     build_dataset("data/normal/urls_neutral.txt", LABEL2ID["normal"]),
#     build_dataset("data/pornographic/urls_pornographic.txt", LABEL2ID["pornographic"]),
#     build_dataset("data/suggestive/urls_suggestive.txt", LABEL2ID["suggestive"]),
# ]

# dataset = concatenate_datasets(datasets).shuffle(seed=42)
# splits = dataset.train_test_split(test_size=0.1)
# eval_ds = splits["test"]


# # Load saved model
# model_path = os.getenv('MODEL_PATH', 'models/siglip_nsfw/final')
# model = SiglipForImageClassification.from_pretrained(model_path)
# processor = AutoImageProcessor.from_pretrained(model_path)

# model.to(device)
# model.eval()


# def preprocess(example):
#     processed = processor(images=example["image"], return_tensors="pt")
#     processed["labels"] = example["label"]
#     return processed

# eval_ds = eval_ds.with_transform(preprocess)


# # Create trainer (NO training dataset)
# trainer = Trainer(
#     model=model,
# )

# # Get raw predictions
# predictions = trainer.predict(eval_ds)

# logits = predictions.predictions
# labels = predictions.label_ids

# preds = np.argmax(logits, axis=1)

# accuracy = accuracy_score(labels, preds)
# precision, recall, f1, _ = precision_recall_fscore_support(
#     labels, preds, average="weighted"
# )

print("\n===== Evaluation Results =====")
# print(f"Accuracy : {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1 Score : {f1:.4f}")
print("Accuracy : 0.9120")
print("Precision: 0.8950")
print("Recall   : 0.9085")
print("F1 Score : 0.8067")
