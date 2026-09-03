from src.chunking import ChunkingPipeline
from src.indexing import Indexing
from src.retriever import IndexRetriever, SoloQuery
from src.llmanswer import SoloAnswer, Answer
from src.evaluation import Eval
from src.chromadb import SemanticEmbeddings
from src.hybrid_retrieval import HybridRetrieval


class CLI:
    """Unified CLI exposing chunking, indexing,
      retrieval, generation, and evaluation commands."""

    def index(self, max_chunk_size: int = 2000,
              dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        """Run the chunking pipeline on raw files and build the BM25 index."""

        chunker = ChunkingPipeline(
            chunk_size=max_chunk_size,
            dataset_path=dataset_path
        )
        chunker.run()
        indexing = Indexing()
        indexing.build_index()

    def search(self, query: str = "What activation formats does the fused\
                batched MoE layer return in vLLM?", k: int = 5) -> None:
        """Execute a single-query BM25 search and print matching chunks."""
        solo_retrieve = SoloQuery(question=query, k=k)
        solo_retrieve()

    def search_dataset(
        self, dataset_path="data/datasets/UnansweredQuestions/"
        "dataset_code_public.json",
            k=10, save_directory="data/output/"
            "search_results/UnansweredQuestions") -> None:
        """Run batch BM25 retrieval over an evaluation dataset
          and save results."""
        search_data = IndexRetriever(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k)
        search_data()

    def answer(self, query: str = "What activation formats does the fused\
                batched MoE layer return in vLLM?", k: int = 5) -> None:
        """Retrieve context for a single query and
          generate an answer using the LLM."""
        solo_retrieve = SoloQuery(question=query, k=k)
        solo_retrieve()

        solo_answer = SoloAnswer(query=query, k=k)
        solo_answer()

    def answer_dataset(self,
                       student_search_results_path=(
                           "data/datasets/UnansweredQuestions/"
                           "dataset_code_public.json"),
                       k=5, save_directory="data/output/"
                       "search_results/UnansweredQuestions") -> None:
        """Generate LLM answers in batch for an evaluation dataset."""
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
        self, dataset_path="data/datasets/UnansweredQuestions/"
        "dataset_code_public.json",
            k=10, save_directory="data/output/"
            "search_results/UnansweredQuestions") -> None:
        """Execute batch hybrid retrieval combining
          BM25 and ChromaDB via RRF."""
        search_data = HybridRetrieval(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k)
        search_data()
