from typing import Any
from pathlib import Path

import bm25s
import json


class Indexing:
    def __init__(self, k=10) -> None:
        self.corpus = []
        self.chunks = []
        self.question_str = []
        self.question_object = []
        self.k = k

    def __call__(self) -> Any:
        self.build_index()

    def build_index(self) -> None:
        list_dict_index = []
        try:

            with open("./data/intern_output/chunk_data.jsonl",
                      "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.corpus.append(data["text"])
                    self.chunks.append(data)
        except Exception:
            print("no chunk_data.jsonl")

        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        try:
            doc_question_path = (
                "./data/dataset/UnansweredQuestions/dataset_docs_public.json")
            code_question_path = (
                "./data/dataset/UnansweredQuestions/dataset_code_public.json")

        except Exception as e:
            print("\n--------\n")
            print(e)

        with open(code_question_path, "r", encoding="UTF-8") as f:
            question = json.load(f)
        for q in question["rag_questions"]:
            query = q["question"]
            query_tokens = bm25s.tokenize(query)

            results, scores = retriever.retrieve(query_tokens, k=self.k)
            # print(results)
            # print(scores)
            retrieved_sources = []
            for index in results[0]:
                start = self.chunks[index]["start_index"]
                end = self.chunks[index]["end_index"]

                retrieved_sources.append(
                    {
                        "file_path": (self.chunks[index]
                                      ["metadata"]["file_path"]),
                        "first_character_index": start,
                        "last_character_index": end

                    })
                dict_json = {
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "retrieved_sources": retrieved_sources

                }
                list_dict_index.append(dict_json)
        final_json = {
            "search_results": list_dict_index,
            "k": self.k
        }
        # print(f"Score: {score:.4f} | Document: {doc}")
        output_path = Path(
            "./data/output/search_results/UnansweredQuestions/index.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="UTF-8") as f:
            json.dump(final_json, f, indent=2)
