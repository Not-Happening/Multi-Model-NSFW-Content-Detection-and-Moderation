import torch
from transformers import SiglipForImageClassification, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import random

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASE_MODEL = "google/siglip-base-patch16-256"
MODEL_PATH = "models/siglip_nsfw/final"
TEST_FILE = "test_images/test.txt"
FACTS_FILE = "facts.txt"

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

def load_facts(filepath):
    """Load facts from file"""
    with open(filepath, 'r') as f:
        facts = [line.strip() for line in f if line.strip()]
    return facts

def apply_black_overlay(image, fact):
    """Apply black overlay to cover the image with a random fact"""
    black_overlay = Image.new('RGB', image.size, color='black')
    draw = ImageDraw.Draw(black_overlay)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default()
    
    # Wrap text to fit on image
    max_width = image.size[0] - 40
    words = fact.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw text in center
    y_offset = (image.size[1] - len(lines) * 40) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (image.size[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y_offset + i * 40), line, fill='white', font=font)
    
    return black_overlay

def run_inference():
    print(f"Using device: {DEVICE}")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    with open(TEST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    facts = load_facts(FACTS_FILE)

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

            if label != "Normal":
                random_fact = random.choice(facts)
                black_image = apply_black_overlay(image, random_fact)
                black_image.show()

        except Exception as e:
            print(f"[{idx}] Failed")
            print(f"     {url}")
            print(f"     Error: {e}")

if __name__ == "__main__":
    run_inference()
