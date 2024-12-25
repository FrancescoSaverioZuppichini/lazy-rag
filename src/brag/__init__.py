from typing import Optional

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
from typing import TypedDict, Union
from PIL import Image
from .logger import logger

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def embed(
    model: ColIdefics3,
    processor: ColIdefics3Processor,
    file_path: str,
    batch_size: Optional[int] = 1,
) -> Union[list[torch.Tensor], list[Image.Image]]:
    images = convert_from_path(file_path, thread_count=4)
    embeddings = None
    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x).to(model.device),
    )
    if device != model.device:
        model.to(device)

    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        embeddings = (
            embeddings_doc
            if embeddings is None
            else torch.vstack([embeddings, embeddings_doc])
        )

    return embeddings, images


def retrieve(
    model: ColIdefics3,
    processor: ColIdefics3Processor,
    query: str,
    embeddings: torch.Tensor,
    images: list[Image.Image],
    k: int = 4,
):
    if device != model.device:
        model.to(device)

    with torch.no_grad():
        batch_query = processor.process_queries(query).to(model.device)
        embeddings_query = model(**batch_query)

    scores = processor.score(embeddings_query, embeddings, device=device)

    top_k_values, top_k_indices = scores.topk(k, dim=1)
    print(top_k_values, top_k_indices.shape)

    results = [
        [
            (images[idx], value, idx)
            for idx, value in zip(indices.tolist(), values.tolist())
        ]
        for indices, values in zip(top_k_indices.cpu(), top_k_values.cpu())
    ]

    return results
