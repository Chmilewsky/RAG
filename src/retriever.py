import bm25s
from pathlib import Path
from src.models import (MinimalSearchResults, MinimalSource,
                        RagDataset, StudentSearchResults)
import json
import Stemmer


class IndexRetriever:
    def __init__(self, dataset_path="data/datasets/UnansweredQuestions/"
                 "dataset_code_public.json",
                 k=10, save_directory="data/output/"
                 "search_results/UnansweredQuestions") -> None:
        self.dataset = Path(dataset_path)
        self.k = k
        self.save_path = Path(save_directory)
        with open("data/intern_output/chunk_data.jsonl",
                  "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        self.import_retriever = bm25s.BM25.load("./data/processed")

    def __call__(self) -> None:
        self.retriever()

    def importcheck(self) -> bool | None:
        with open(self.dataset, "r", encoding="UTF-8") as f:
            check = RagDataset.model_validate_json(f.read())
        if check:
            return True

    def retriever(self):
        search_results = []
        stemmer = Stemmer.Stemmer("english")

        with open(self.dataset, "r", encoding="UTF-8") as f:
            question = RagDataset.model_validate_json(f.read())

        for q in question.rag_questions:
            query = q.question
            query_tokens = bm25s.tokenize(query, stemmer=stemmer)

            results, scores = self.import_retriever.retrieve(
                query_tokens, k=self.k)
            retrieved_sources = []
            for index in results[0]:
                start = self.chunks[index]["start_index"]
                end = self.chunks[index]["end_index"]

                retrieved_sources.append(
                    MinimalSource(
                        file_path=(self.chunks[index]
                                   ["metadata"]["file_path"]),
                        first_character_index=start,
                        last_character_index=end,
                        chunk_txt=self.chunks[index]["text"]
                    ))

            search_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=retrieved_sources
                ))

        final_output = StudentSearchResults(
            search_results=search_results, k=self.k)

        savefile = self.save_path / self.dataset.name
        savefile.parent.mkdir(parents=True, exist_ok=True)
        savefile.write_text(
            final_output.model_dump_json(
                indent=2), encoding="UTF-8")
        print(f"Retrieve completed\nFile at: {savefile}")


class SoloQuery:
    def __init__(
            self, question="what is the answer to all question", k=5) -> None:
        self.save_path = Path("data/output/search_results/UnansweredQuestions")
        self.question = question
        self.k = k
        with open("data/intern_output/chunk_data.jsonl",
                  "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.import_retriever = bm25s.BM25.load("./data/processed")

    def __call__(self) -> None:
        self.retriever()

    def retriever(self):
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(self.question, stemmer=stemmer)
        results, scores = self.import_retriever.retrieve(
            query_tokens, k=self.k)
        print(f"Question : {self.question} ")
        search_results = []
        retrieved_sources = []
        for index in results[0]:
            start = self.chunks[index]["start_index"]
            end = self.chunks[index]["end_index"]

            retrieved_sources.append(
                MinimalSource(
                    file_path=(self.chunks[index]
                               ["metadata"]["file_path"]),
                    first_character_index=start,
                    last_character_index=end,
                    chunk_txt=self.chunks[index]["text"]
                ))

        search_results.append(
            MinimalSearchResults(
                question_id="01",
                question=self.question,
                retrieved_sources=retrieved_sources
            ))

        final_output = StudentSearchResults(
            search_results=search_results, k=self.k)
        for source in final_output.search_results:
            for output in source.retrieved_sources:
                print(
                    f"{output.file_path} "
                    f"[{output.first_character_index}:"
                    f"{output.last_character_index}]")

        savefile = self.save_path / "solo_answer.json"
        savefile.parent.mkdir(parents=True, exist_ok=True)
        savefile.write_text(
            final_output.model_dump_json(
                indent=2), encoding="UTF-8")
        print(f"\n---Retrieve completed---\nFile at: {savefile}")
