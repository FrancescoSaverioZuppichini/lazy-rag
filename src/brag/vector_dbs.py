from typing import TypedDict
from qdrant_client.models import PointStruct
from dataclasses import dataclass

class DocumentMetada(TypedDict):
    mime_type: str
    page_number: int = 0

class Document:
    vector: list[int]
    metadata:

class VectorDb:
    def insert(self, points: ):
        pass
    