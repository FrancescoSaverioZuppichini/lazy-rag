
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict
from PIL import Image

MIME_TYPE = Literal[
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "application/pdf",
    ] 

class EmbeddingSource(TypedDict):
    name: Optional[str]
    mime_type: MIME_TYPE
    content: str | Image.Image | list[int] | None

@dataclass
class Embedding:
    vector: list[float]
    source: EmbeddingSource