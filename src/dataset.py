import requests
from PIL import Image
from io import BytesIO
from datasets import Dataset
from tqdm import tqdm

def build_dataset(txt_path: str, label: int, max_images: int = 300):
    with open(txt_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    urls = urls[:max_images]

    images = []
    labels = []

    for url in tqdm(urls, desc=f"Label {label}"):
        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
            labels.append(label)
        except Exception:
            continue

    return Dataset.from_dict({
        "image": images,
        "label": labels
    })
