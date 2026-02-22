import torch
from transformers import SiglipForImageClassification, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASE_MODEL = "google/siglip-base-patch16-256"
MODEL_PATH = "models/siglip_nsfw/final"
TEST_FILE = "test_images/test.txt"

LABEL_MAP = {
    0: "Anime Picture",
    1: "Hentai",
    2: "Normal",
    3: "Pornography",
    4: "Enticing or Sensual"
}

def load_image_from_url(url, timeout=10):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def run_inference():
    print(f"Using device: {DEVICE}")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    with open(TEST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"\nFound {len(urls)} test image URLs\n")

    for idx, url in enumerate(urls, 1):
        try:
            image = load_image_from_url(url)

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            pred_id = probs.argmax(dim=-1).item()
            confidence = probs[0][pred_id].item()
            label = LABEL_MAP[pred_id]

            print(f"[{idx}] {label} | confidence={confidence:.4f}")
            print(f"     {url}")

            if label != "Normal" :
                image.show()

        except Exception as e:
            print(f"[{idx}] Failed")
            print(f"     {url}")
            print(f"     Error: {e}")

if __name__ == "__main__":
    run_inference()
