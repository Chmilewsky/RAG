from typing import Any
from tqdm import tqdm
import Stemmer
import bm25s
import json


class Indexing:
    """Build and persist a BM25 index from pre-extracted JSONL chunk data."""

    def __init__(self) -> None:
        """Initialize an empty corpus list."""
        self.corpus = []

    def __call__(self) -> Any:
        """Execute the index building pipeline."""
        self.build_index()

    def build_index(self) -> None:
        """Load chunks, tokenize enriched corpus text,
          compute the BM25 index, and save it to disk."""
        stemmer = Stemmer.Stemmer("english")
        count_chunk = 0

        with open("./data/intern_output/chunk_data.jsonl",
                  "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="tokenizing"):
                data = json.loads(line)
                text = data['text']
                filename = data['metadata']['filename']
                filepath = data['metadata']['file_path']
                text_expanded = text.replace("_", " ")
                corpus_str = f"{text} {text_expanded} {filename} {filepath}"
                self.corpus.append(corpus_str)
                count_chunk += 1

        corpus_tokens = bm25s.tokenize(
            self.corpus, stemmer=stemmer, stopwords="en")
        retriever = bm25s.BM25(k1=0.4, b=0.4)
        retriever.index(corpus_tokens)
        retriever.save(str("data/processed"))
        print(
            f"Ingestion complete! Indexed {count_chunk} under data/processed")
