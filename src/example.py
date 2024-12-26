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
    attn_implementation="flash_attention_2",
    # attn_implementation="eager",
).eval()
processor = ColIdefics3Processor.from_pretrained("vidore/colsmolvlm-alpha")

embeddings, images = embed(
    model,
    processor,
    "https://www.zurich.ch/-/media/zurich-site/content/privatkunden/wohnen-bauen/dokumente/factsheet-all-risk/zurichfactsheetallrisk.pdf",
    batch_size=8,
)

print(embeddings[0].shape)
embeddings, images = embed(model, processor, "foo.jpg", batch_size=8)

print(embeddings[0].shape)
embeddings, images = embed(
    model, processor, "examples/boring_insurance.pdf", batch_size=8
)
print(embeddings[0].shape)

from brag import embed_text

texts = [
    "Napoleon was a great general",
    "Past is an Italian food",
    "Audio RS is a shitty car",
    "Francesco is a Machine Learning Engineer",
]


query = ["looking for something about war", "who is Francesco?"]

results = retrieve(model, processor, query, embeddings, texts, 2)

print(results)
