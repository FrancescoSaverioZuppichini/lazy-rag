from abc import ABC, abstractmethod
from .const import COLLECTION_NAME
from .schemas import Embedding

class DB(ABC):
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = COLLECTION_NAME

    @abstractmethod
    def insert(self, embeddings: list[Embedding]):
        pass
        
    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def search(self):
        pass
