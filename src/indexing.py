from typing import Any
from pathlib import Path

import bm25s
import json

# ajouter stemmer et stopwords


class Indexing:
    def __init__(self) -> None:
        self.corpus = []

    def __call__(self) -> Any:
        self.build_index()

    def build_index(self) -> None:
        # list_dict_index = []
        try:

            with open("./data/intern_output/chunk_data.jsonl",
                      "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.corpus.append(data["text"])

        except Exception:
            print("no chunk_data.jsonl")

        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str("data/processed"))
