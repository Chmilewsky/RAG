from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from typing import Any
from tqdm import tqdm
from collections.abc import Iterator
import json
from src.models import ChunkMetadata, ChunkModel


class ChunkingPipeline:
    """Orchestrates scanning, chunking, and writing."""

    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        """Initialize the chunking pipeline components.

        Args:
            chunk_size: Target maximum size per chunk.
            dataset_path: Path to the dataset folder or file.
        """
        self.output_path: Path = Path(
            "./data/intern_output/chunk_data_lang.jsonl")
        self.dataset_path = Path(dataset_path)
        if self.output_path.exists():
            self.output_path.unlink()

        self.scanner = FileScanner(dataset_path)
        self.chunker = FileChunker(chunk_size=chunk_size)
        self.writer = JsonWriter(output_path=self.output_path)

    def __call__(self, *args: Any, **kwds: Any) -> None:
        """Allow the instance to be called directly to start processing."""
        self.run()

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
                chunks = self.chunker.file_type_filter(file)
            except Exception as e:
                raise e
            if chunks:
                self.writer.write(chunks)


class FileScanner:
    """Scans and yields files from a directory or single path."""

    def __init__(self, dataset_path: str) -> None:
        """Initialize the scanner with a dataset path."""
        self.data_path = dataset_path

    def folder_or_file(self) -> Iterator:
        """Yield valid file paths from the dataset directory or single file.

        Yields:
            Path: Next file path to process.
        """
        p = Path(self.data_path)
        try:
            if not p.exists():
                print(f"\nno file or folder at {p.absolute()}\n")
            elif p.is_dir():
                for item in p.rglob("*"):
                    if item.is_file():
                        yield item
            elif p.is_file():
                yield p
        except PermissionError as e:
            raise e


class FileChunker:
    """Applies appropriate chunking strategies based on file types."""

    def __init__(self, chunk_size: int = 2000) -> None:
        """Initialize chunker with maximum size and fallback options."""
        self.chunk_size = chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size, chunk_overlap=chunk_size // 10,
            add_start_index=True
        )

    def file_type_filter(self, data) -> list[dict[str, Any]] | None:
        """Filter file by extension and route to the corresponding chunker.

            Args:
                data: Path to the target file.

            Returns:
                List of chunk dictionaries,
                or None if file type is unsupported.
            """
        if data.suffix == ".py":
            chunks = self.langchunk(data)
            return chunks

    def langchunk(self, data):
        with open(data, "r", encoding="UTF-8") as f:
            data_txt = f.read()
        chunks = (self.text_splitter.create_documents
                  (texts=[data_txt],
                   metadatas=[{"file_path": data}])
                  )
        formated_chunk = []
        for chunk in chunks:
            starting_index = chunk.metadata["start_index"]
            ending_index = (
                chunk.metadata["start_index"]
                + len(chunk.page_content))
            acces = Path(data)
            chunk_meta = ChunkMetadata(
                filename=acces.name,
                file_path=str(acces)
            )
            chunk_final = ChunkModel(
                text=chunk.page_content,
                start_index=starting_index,
                end_index=ending_index,
                token_count=len(chunk.page_content),
                metadata=chunk_meta
            )
            formated_chunk.append(chunk_final)
        return formated_chunk


class JsonWriter:
    """Appends processed chunk dictionaries to a JSONL output file."""

    def __init__(self, output_path) -> None:
        """Initialize writer with target output path."""
        self.output_path = output_path

    def write(self, chunks):
        """Append a list of chunk dictionaries to the JSONL output file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "a", encoding="UTF-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")


def main():
    test = ChunkingPipeline(2000, "./data/raw/vllm-0.10.1")
    test()


if __name__ == "__main__":
    main()
