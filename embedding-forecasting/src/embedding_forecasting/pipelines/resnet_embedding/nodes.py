import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.IMAGENET1K_V2
base = resnet50(weights=weights)
resnet_model = torch.nn.Sequential(*list(base.children())[:-1]).to(device)
resnet_model.eval()
resnet_transform = weights.transforms()

def embed_resnet(img: Image.Image) -> np.ndarray:
    tensor = resnet_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet_model(tensor)
    return feat.squeeze().cpu().numpy()

def embed_resnet_list(images: list[Image.Image]) -> list[np.ndarray]:
    return [embed_resnet(img) for img in images]
