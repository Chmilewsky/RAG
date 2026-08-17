from typing import Any
from tqdm import tqdm
import Stemmer

import bm25s
import json

# ajouter stemmer et stopwords


class Indexing:
    def __init__(self) -> None:
        self.corpus = []

    def __call__(self) -> Any:
        self.build_index()

    def build_index(self) -> None:
        stemmer = Stemmer.Stemmer("english")
        count_chunk = 0
        try:
            with open("./data/intern_output/chunk_data.jsonl",
                      "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="tokenizing"):
                    data = json.loads(line)
                    corpus_str = (f"{data["text"]}"
                                  f"{data['metadata']['filename']}")
                    self.corpus.append(corpus_str)
                    count_chunk += 1

        except Exception as e:
            print(f"no chunk_data.jsonl --\n{e}\n----")
            print(e)

        corpus_tokens = bm25s.tokenize(
            self.corpus, stemmer=stemmer, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(str("data/processed"))
        print(
            f"Ingestion complete! Indexed {count_chunk} under data/processed")
