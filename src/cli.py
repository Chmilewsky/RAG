from src.chunking import ChunkingPipeline
from src.indexing import Indexing
from src.retriever import IndexRetriever, SoloQuery
from src.llmanswer import SoloAnswer, Answer
from src.evaluation import Eval


class CLI:
    def index(self, max_chunk_size: int = 2000,
              dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        chunker = ChunkingPipeline(
            chunk_size=max_chunk_size,
            dataset_path=dataset_path
        )
        chunker.run()
        indexing = Indexing()
        indexing.build_index()

    def search(self, query: str = "what is the answer to all question",
               k: int = 5) -> None:
        solo_retrieve = SoloQuery(question=query, k=k)
        solo_retrieve()

    def search_dataset(
        self, dataset_path="data/datasets/UnansweredQuestions/"
        "dataset_code_public.json",
            k=10, save_directory="data/output/"
            "search_results/UnansweredQuestions") -> None:
        search_data = IndexRetriever(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k)
        search_data()

    def answer(self, query: str = "how to configure the OpenAI server?",
               k: int = 5) -> None:
        solo_answer = SoloAnswer(query=query, k=k)
        solo_answer()

    def answer_dataset(self,
                       student_search_results_path=(
                           "data/datasets/UnansweredQuestions/"
                           "dataset_code_public.json"),
                       k=5, save_directory="data/output/"
                       "search_results/UnansweredQuestions") -> None:
        answer_data = Answer(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory)
        answer_data()

    def evaluate(self,
                 student_search_results_path: str =
                 ("data/output/search_results"
                  "/UnansweredQuestions/dataset_code_public.json"),
                 dataset_path: str =
                 ("data/datasets/AnsweredQuestions/dataset_code_public.json")):
        eval = Eval(student_search_results_path=student_search_results_path,
                    dataset_path=dataset_path)
        eval()
