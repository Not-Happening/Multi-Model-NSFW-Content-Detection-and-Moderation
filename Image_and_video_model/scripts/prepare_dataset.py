from src.dataset import build_dataset
from datasets import concatenate_datasets

LABEL2ID = {
    "drawings": 0,
    "hentai": 1,
    "normal": 2,
    "pornographic": 3,
    "suggestive": 4,
}

datasets = [
    build_dataset("data/drawings/urls_drawings.txt", LABEL2ID["drawings"]),
    build_dataset("data/hentai/urls_hentai.txt", LABEL2ID["hentai"]),
    build_dataset("data/normal/urls_neutral.txt", LABEL2ID["normal"]),
    build_dataset("data/pornographic/urls_pornographic.txt", LABEL2ID["pornographic"]),
    build_dataset("data/suggestive/urls_suggestive.txt", LABEL2ID["suggestive"]),
]

full_dataset = concatenate_datasets(datasets).shuffle(seed=42)

print(full_dataset)
print("Samples per label:")
print(full_dataset.features)

