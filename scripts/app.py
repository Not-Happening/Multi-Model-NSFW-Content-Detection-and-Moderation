import os
import torch
import random
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from transformers import SiglipImageProcessor, SiglipForImageClassification

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load model and processor
MODEL_PATH = os.getenv('MODEL_PATH', 'models/siglip_nsfw/final')
BASE_MODEL = "google/siglip-base-patch16-256"
device = "cpu"  # Railway uses CPU

print(f"Loading model from {MODEL_PATH}...")
processor = SiglipImageProcessor.from_pretrained(BASE_MODEL)
model = SiglipForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Load facts
with open('facts.txt', 'r') as f:
    facts = [line.strip() for line in f if line.strip()]

LABEL_MAP = {
    0: "Anime Picture",
    1: "Hentai",
    2: "Normal",
    3: "Pornography",
    4: "Enticing or Sensual"
}

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        data = request.json
        image_url = data.get('imageUrl')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # Download image from URL
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Process and classify
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        confidence = float(torch.softmax(logits, dim=-1).max().item())
        
        # Check if NSFW (class 2 is "Normal")
        is_nsfw = predicted_class != 2
        random_fact = random.choice(facts)
        label = LABEL_MAP.get(predicted_class, "Unknown")
        
        return jsonify({
            'isNSFW': is_nsfw,
            'fact': random_fact,
            'confidence': confidence,
            'label': label
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0')