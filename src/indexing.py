from typing import Any

import bm25s
import json


class Indexing:
    def __init__(self) -> None:
        self.corpus = []
        self.chunks = []

    def __call__(self) -> Any:
        self.build_index()

    def build_index(self) -> None:
        try:

            with open("chunk_data.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.corpus.append(data["text"])
                    self.chunks.append(data)

            corpus_tokens = bm25s.tokenize(self.corpus)
            retriever = bm25s.BM25(corpus=self.corpus)
            retriever.index(corpus_tokens)
            for i in retriever.scores:
                print(i)

            query = "server"
            query_tokens = bm25s.tokenize(query)

            results, scores = retriever.retrieve(query_tokens, k=3)

            for doc, score in zip(results[0], scores[0]):
                print(f"Score: {score:.4f} | Document: {doc}")
        except Exception:
            print("no chunk_data.jsonl")
