import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Environment ready ✅")
