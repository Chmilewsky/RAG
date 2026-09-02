from typing import Any
from chonkie import SemanticChunker
from collections.abc import Iterator
from pathlib import Path
from src.chunking import FileScanner, FileChunker, JsonWriter
from functools import lru_cache
from tqdm import tqdm


class Sematic:
    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        self.output_path: Path = Path(
            "./data/intern_output/chunk_data_semantic.jsonl")
        self.dataset_path = Path(dataset_path)
        self.chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-32M",
            threshold=0.8,
            chunk_size=chunk_size,
            similarity_window=3,
            skip_window=0
        )
        self.scanner = FileScanner(dataset_path)
        self.chunker = FileChunker(chunk_size=chunk_size)
        self.writer = JsonWriter(output_path=self.output_path)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.run()

    @lru_cache(maxsize=None)
    def run(self) -> None:
        """Run the full chunking pipeline on all target files."""
        p = self.dataset_path
        total_files = 1
        if p.is_dir():
            total_files = 0
            for item in p.rglob("*"):
                if item.is_file():
                    total_files += 1

        for file in tqdm(self.scanner.folder_or_file(),
                         desc="chunking files", total=total_files):
            try:
                chunks = self.chunker.semantic_chonking(file)
                if chunks:
                    self.writer.write(chunks)
            except Exception as e:
                print(f"file {e} coulnt be loaded")
