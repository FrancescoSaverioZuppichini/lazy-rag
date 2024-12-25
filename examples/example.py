from brag import embed, retrieve
import os

import torch

from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import (
    ColQwen2,
    ColQwen2Processor,
    ColIdefics3,
    ColIdefics3Processor,
)

model = ColIdefics3.from_pretrained(
    "vidore/colsmolvlm-alpha",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",  # or eager
).eval()
processor = ColIdefics3Processor.from_pretrained("vidore/colsmolvlm-alpha")

embeddings, images = embed(model, processor, "examples/boring_insurance.pdf")
print(embeddings.shape)


