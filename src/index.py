from typing import Any

from src.chunking import ChunkingPipeline
from src.indexing import Indexing
from src.retriever import IndexRetriever, SoloQuery
from src.llmanswer import SoloAnswer, Answer


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


class Search:
    def __init__(
        self, question="how to configure the OpenAI server?",
        k=5, save_directory="data/output/"
            "search_results/UnansweredQuestions") -> None:
        self.question = question
        self.k = k
        self.save = save_directory

    def __call__(self) -> None:
        solo_retrieve = SoloQuery(question=self.question, k=self.k)
        solo_retrieve()


class SearchDataset:

    def __init__(
        self, dataset_path="data/datasets/UnansweredQuestions/"
        "dataset_code_public.json",
            k=10, save_directory="data/output/"
            "search_results/UnansweredQuestions") -> None:
        self.dataset = dataset_path
        self.k = k
        self.save = save_directory

    def __call__(self) -> None:
        """default action"""
        index_retreive = IndexRetriever(dataset_path=self.dataset, k=self.k,
                                        save_directory=self.save)
        index_retreive()


class AnswerDataset:
    def __init__(self, question="how to configure the OpenAI server?",
                 k=5, save_directory="data/output/"
                 "search_results/UnansweredQuestions") -> None:
        self.question = question
        self.k = k
        self.save = save_directory

    def __call__(self) -> None:
        answer = Answer(question_file=self.question, k=self.k)
        answer()


class SoloAnswer:
    def __init__(
            self, query="how to configure the OpenAI server?", k=5) -> None:
        self.question = query
        self.k = k

    def __call__(self) -> None:
        solo_answer = SoloAnswer(query=self.question, k=self.k)
        solo_answer()
