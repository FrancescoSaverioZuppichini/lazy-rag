import base64
from PIL import Image
import torch
import torch.nn.functional as F

def encode_image(image: Image) -> str:
    return base64.b64encode(image.tobytes()).decode('utf-8')

def pad_sequence(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(emb.shape[1] for emb in tensors)
    padded = torch.cat(
        [F.pad(emb, (0, 0, 0, max_len - emb.shape[1])) for emb in embeddings]
    )
    return padded