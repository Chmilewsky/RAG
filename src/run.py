from src.chunking import FileChunker
from src.indexing import Indexing


class Run:
    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./vllm-0.10.1") -> None:
        self.data_path = dataset_path
        self.chunk_size = chunk_size

    def __call__(self) -> None:
        """Default action"""
        chunker = FileChunker(
            chunk_size=self.chunk_size,
            dataset_path=self.data_path)
        chunker()
        # indexing = Indexing()
        # indexing()
