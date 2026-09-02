import Stemmer
from pathlib import Path
import json
from src.models import (MinimalSearchResults, MinimalSource,
                        RagDataset, StudentSearchResults)
import bm25s
import chromadb


class HybridRetrieval:
    def __init__(self, dataset_path="data/datasets/UnansweredQuestions/"
                 "dataset_code_public.json",
                 k=10, save_directory="data/output/"
                 "search_results/UnansweredQuestions") -> None:
        self.dataset = Path(dataset_path)
        self.k = k
        self.save_path = Path(save_directory)
        client = chromadb.PersistentClient(
            path="data/intern_output/vector_DataBase")
        self.collection = client.get_collection(name="my_collection")
        with open("data/intern_output/chunk_data.jsonl",
                  "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.chunks_by_id = {chunk["id"]: chunk for chunk in self.chunks}

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
            retrieved_sources = []

            results, scores = self.import_retriever.retrieve(
                query_tokens, k=self.k)
            db_result = self.collection.query(
                query_texts=[query], n_results=self.k)
            top_k = self.rrf(results[0], db_result["ids"][0])
            for k in top_k[:self.k]:
                chunk = self.chunks_by_id[k]
                start = chunk["start_index"]
                end = chunk["end_index"]
                retrieved_sources.append(
                    MinimalSource(
                        file_path=(chunk
                                   ["metadata"]["file_path"]),
                        first_character_index=start,
                        last_character_index=end,
                        chunk_txt=chunk["text"]
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

    def rrf(self, bm25, db):
        k = 60
        rrf_scores = {}
        for rank, index in enumerate(bm25, start=1):
            id = self.chunks[index]["id"]
            rrf_scores[id] = rrf_scores.get(id, 0.0) + (1.0 / (k + rank))
        for rank, id in enumerate(db, start=1):
            rrf_scores[id] = rrf_scores.get(id, 0.0) + (1.0 / (k + rank))

        sorted_results = sorted(
            rrf_scores,
            key=lambda x: rrf_scores[x],
            reverse=True)
        return sorted_results
