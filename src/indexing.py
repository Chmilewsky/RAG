from typing import Any
from pathlib import Path

import bm25s
import json


class Indexing:
    def __init__(self, k=10) -> None:
        self.corpus = []
        self.chunks = []
        self.k = k

    def __call__(self) -> Any:
        self.build_index()

    def build_index(self) -> None:
        try:

            with open("chunk_data.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.corpus.append(data["text"])
                    self.chunks.append(data)
        except Exception:
            print("no chunk_data.jsonl")

        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        query = "server"
        query_tokens = bm25s.tokenize(query)

        results, scores = retriever.retrieve(query_tokens, k=self.k)
        print(results)
        print(scores)

        for index, score in zip(results[0], scores[0]):
            print(f"{self.chunks[index]["text"]}")
            dict_json = {"text": self.chunks[index]["text"],
                         "question_id": "xx",
                         "question": "xx",
                         "answer": "xx",
                         "sources": [
                {
                    "file_path": self.chunks[index]["metadata"]["file_path"],
                    "first_character_index": self.chunks[index]["start_index"],
                    "last_character_index": self.chunks[index]["end_index"]
                }
            ],


            }
            # print(f"Score: {score:.4f} | Document: {doc}")
        output_path = Path("./data/processed/index.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="UTF-8") as f:
            f.write(json.dumps(dict_json, ensure_ascii=False) + "\n")
