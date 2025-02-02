from dataclasses import dataclass
import multiprocessing
from pathlib import Path, PurePath
from typing import Literal, Optional, TypedDict, Union
from pdf2image import convert_from_bytes, convert_from_path
import requests
from PIL import Image
from .logger import logger
from abc import ABC, abstractmethod

MIME_TYPE = Literal[
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "application/pdf",
    ] 

class Document:
    mime_type: MIME_TYPE = None

    @abstractmethod
    def __enter__(self) -> list[Image.Image]:
        pass


SUPPORTED_MIME_TYPES = {
    "image/jpeg": (".jpg", ".jpeg"),
    "image/png": ".png",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": (".tif", ".tiff"),
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}

SUPPORTED_FORMATS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


@dataclass
class URLDocument(Document):
    url: str
    headers: Optional[dict[str, str]] = None
    mime_type: MIME_TYPE = None
    images: list[Image.Image] = None

    def __enter__(self) -> list[Image.Image]:
        res = requests.get(self.url, headers=self.headers, stream=True)
        res.raise_for_status()
        res.raw.decode_content = True
        # set it to default
        if self.mime_type is None:
            self.mime_type = res.headers.get("Content-Type", None)
            if self.mime_type not in SUPPORTED_MIME_TYPES:
                raise ValueError(
                    (
                        f"Unsupported content-type from server: {self.mime_type}: if you know what it is please pass `mime_type` in the constructor."
                    )
                )
        media_type = self.mime_type.split("/")[0]
        if media_type == "image":
            self.images = [Image.open(res.raw)]
        else:
            self.images = convert_from_bytes(
                res.raw.data, thread_count=multiprocessing.cpu_count()
            )
        return self

    def __exit__(self, *args):
        pass


@dataclass
class FileDocument(Document):
    path: Union[str, PurePath]
    images: list[Image.Image] = None

    def __enter__(self) -> list[Image.Image]:
        path = self.path
        if not isinstance(path, PurePath):
            path = PurePath(path)
        doc_format = path.suffix

        if doc_format not in SUPPORTED_FORMATS:
            raise ValueError((f"Unsupported file format: {doc_format}"))

        self.mime_type = SUPPORTED_FORMATS[doc_format]
        if doc_format == ".pdf":
            self.images = convert_from_path(
                path, thread_count=multiprocessing.cpu_count()
            )
        else:
            self.images = [Image.open(path)]

        return self

    def __exit__(self, *args):
        pass


