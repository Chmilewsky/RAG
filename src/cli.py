from src.chunking import ChunkingPipeline
from src.indexing import Indexing
from src.retriever import IndexRetriever, SoloQuery
from src.llmanswer import SoloAnswer, Answer
from src.evaluation import Eval
from src.chromadb import SemanticEmbeddings
from src.hybrid_retrieval import HybridRetrieval
from src.ollama_server import OllamaService
import sys


class CLI:
    """Unified CLI exposing chunking, indexing,
      retrieval, generation, and evaluation commands."""

    def index(self, max_chunk_size: int = 2000,
              dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        """Run the chunking pipeline on raw files and build the BM25 index."""
        if not 200 <= max_chunk_size <= 2000:
            print("Max chunk size must be between 200 and 2000")
            sys.exit(1)
        if not dataset_path.strip():
            print("path for data set cant be empty")
            sys.exit(1)
        try:
            chunker = ChunkingPipeline(
                chunk_size=max_chunk_size,
                dataset_path=dataset_path
            )
            chunker.run()
            indexing = Indexing()
            indexing.build_index()
        except FileNotFoundError as e:
            print(f"File or directory not found: {e}", file=sys.stderr)
            sys.exit(1)

    def search(self, query: str = "What activation formats does the fused\
                batched MoE layer return in vLLM?", k: int = 5) -> None:
        """Execute a single-query BM25 search and print matching chunks."""
        if k <= 0:
            print("k must be superior to 0")
            sys.exit(1)
        if not query.strip():
            print("query cant be empty")
            sys.exit(1)
        try:
            solo_retrieve = SoloQuery(question=query, k=k)
            solo_retrieve()
        except FileNotFoundError as e:
            print(f"File or directory not found: {e}", file=sys.stderr)
            sys.exit(1)

    def search_dataset(
        self, dataset_path: str = "data/datasets/UnansweredQuestions/"
        "dataset_code_public.json",
            k: int = 10, save_directory: str = "data/output/"
            "search_results/UnansweredQuestions") -> None:
        """Run batch BM25 retrieval over an evaluation dataset
          and save results."""
        search_data = IndexRetriever(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k)
        search_data()

    def answer(self, query: str = ("What activation formats does the fused"
                                   "batched MoE layer return in vLLM?"),
               k: int = 5) -> None:
        """Retrieve context for a single query and
          generate an answer using the LLM."""
        service = OllamaService()
        service.start()
        solo_retrieve = SoloQuery(question=query, k=k)
        solo_retrieve()

        solo_answer = SoloAnswer(query=query, k=k)
        solo_answer()

    def answer_dataset(self,
                       student_search_results_path: str = (
                           "data/output/search_results/"
                           "UnansweredQuestions/dataset_code_public.json"),
                       k: int = 5, save_directory: str = "data/output/"
                       "search_results/UnansweredQuestions") -> None:
        """Generate LLM answers in batch for an evaluation dataset."""
        service = OllamaService()
        service.start()
        answer_data = Answer(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory)
        answer_data()

    def evaluate(self,
                 student_search_results_path: str =
                 ("data/output/search_results"
                  "/UnansweredQuestions/dataset_code_public.json"),
                 dataset_path: str =
                 ("data/datasets/AnsweredQuestions"
                  "/dataset_code_public.json")) -> None:
        """Compute Recall@k metrics
          comparing retrieval results against ground truth."""
        eval = Eval(student_search_results_path=student_search_results_path,
                    dataset_path=dataset_path)
        eval()

    def semantic(self, max_chunk_size: int = 2000,
                 dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        """Chunk raw documents, build the BM25 index,
          and populate ChromaDB embeddings."""
        chunker = ChunkingPipeline(
            chunk_size=max_chunk_size,
            dataset_path=dataset_path
        )
        chunker.run()
        indexing = Indexing()
        indexing.build_index()
        semantic = SemanticEmbeddings()
        semantic()

    def hybrid(
        self, dataset_path: str = "data/datasets/UnansweredQuestions/"
        "dataset_code_public.json",
            k: int = 10, save_directory: str = "data/output/"
            "search_results/UnansweredQuestions") -> None:
        """Execute batch hybrid retrieval combining
          BM25 and ChromaDB via RRF."""
        search_data = HybridRetrieval(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k)
        search_data()
