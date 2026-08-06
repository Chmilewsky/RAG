from src.chunking import ChunkingPipeline
from src.indexing import Indexing
from src.retriever import IndexRetriever


class ChunkIndex:
    def __init__(self, max_chunk_size: int = 2000,
                 dataset_path: str = "data/raw/vllm-0.10.1") -> None:
        self.data_path = dataset_path
        self.chunk_size = max_chunk_size

    def __call__(self) -> None:
        """Default action"""
        chunker = ChunkingPipeline(
            chunk_size=self.chunk_size,
            dataset_path=self.data_path)
        chunker()
        indexing = Indexing()
        indexing()


class SearchDataset:

    def __init__(
        self, dataset_path="data/dataset/UnansweredQuestions/"
        "dataset_code_public.json",
            k=10, save_directory="data/output/"
            "search_results/UnansweredQuestions") -> None:
        self.dataset = dataset_path
        self.k = k
        self.save = save_directory

    def __call__(self) -> None:
        """default action"""
        indexretreive = IndexRetriever(dataset_path=self.dataset, k=self.k,
                                       save_directory=self.save)
        indexretreive()
