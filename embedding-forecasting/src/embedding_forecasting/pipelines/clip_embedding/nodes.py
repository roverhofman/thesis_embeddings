import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# device & model init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    device_map="cpu",
    use_safetensors=True
).to(device)
clip_model.eval()

def embed_clip(img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    return feats[0].cpu().numpy()

def embed_clip_list(images: list[Image.Image]) -> list[np.ndarray]:
    """
    Embed a list of PIL images with CLIP, returning a list of numpy arrays.
    """
    return [embed_clip(img) for img in images]
