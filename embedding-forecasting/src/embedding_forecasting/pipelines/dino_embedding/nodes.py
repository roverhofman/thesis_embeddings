import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = (
    AutoModel
    .from_pretrained("facebook/dinov2-base", device_map="cpu", use_safetensors=True)
    .to(device)
)
dino_model.eval()

def embed_dino(img: Image.Image) -> np.ndarray:
    inputs = dino_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = dino_model(**inputs)
    return out.last_hidden_state[0, 0, :].cpu().numpy()

def embed_dino_list(images: list[Image.Image]) -> list[np.ndarray]:
    return [embed_dino(img) for img in images]
