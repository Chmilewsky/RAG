from collections.abc import Iterator
from chonkie import Pipeline, TokenChunker, Chunk, RecursiveRules
from magika import Magika
from pathlib import Path
import json
from typing import Any
from tqdm import tqdm


class ChunkingPipeline:
    """Orchestrates scanning, chunking, and writing."""

    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./data/raw/vllm-0.10.1") -> None:
        """Initialize the chunking pipeline components.

        Args:
            chunk_size: Target maximum size per chunk.
            dataset_path: Path to the dataset folder or file.
        """
        self.output_path: Path = Path("./data/intern_output/chunk_data.jsonl")
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
        self.m = Magika()
        self.tokenchunker = TokenChunker(
            tokenizer="character",
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10
        )
        self.count = 0

    def file_type_filter(self, data) -> list[dict[str, Any]] | None:
        """Filter file by extension and route to the corresponding chunker.

        Args:
            data: Path to the target file.

        Returns:
            List of chunk dictionaries, or None if file type is unsupported.
        """
        # if data.suffix == ".py":
        #     chunks = self.py_chonking(data)
        #     return chunks
        if data.suffix == ".md":
            chunks = self.md_chonking(data)
            return chunks
        # elif data.suffix in [".yaml", ".cu", ".sh", ".toml"]:
        #     chunks = self.magika_chonking(data)
        #     return chunks

    def md_chonking(self, data) -> list[dict[str, Any]]:
        """Chunk a Markdown file using recursive chunking."""
        data_path = str(data)
        with open("src/custom_markdown.json", "r", encoding="utf-8") as f:
            rules_dict = json.load(f)
        custom_rules = RecursiveRules.from_dict(rules_dict)
        md_pipeline = (
            Pipeline().fetch_from(
                "file", path=data_path)
            .process_with("markdown")
            .chunk_with("recursive", chunk_size=self.chunk_size,
                        tokenizer="character", rules=custom_rules).run())

        merge_chunk = self.chunk_merger(
            md_pipeline, 1750, self.chunk_size)

        chunks = self.metadata_add(merge_chunk, data)
        return chunks

    def py_chonking(self, data) -> list[dict[str, Any]]:
        """Chunk a Python file using AST code chunking."""
        data_path = str(data)
        # print(data_path)
        py_pipeline = (
            Pipeline().fetch_from(
                "file", path=data_path)

            .chunk_with("code", language="python", chunk_size=self.chunk_size,
                        tokenizer="character").run())
        chunks = self.metadata_add(py_pipeline, data)
        return chunks

    def magika_chonking(self, data) -> list[dict[str, Any]] | None:
        """Identify file language using Magika and apply code chunking."""
        data_path = str(data)
        res = self.m.identify_path(data_path)
        lang = res.output.label
        try:
            magika_pipeline = (
                Pipeline().fetch_from(
                    "file", path=data_path)

                .chunk_with("code", language=lang,
                            chunk_size=self.chunk_size,
                            tokenizer="character").run())
            chunks = self.metadata_add(magika_pipeline, data)
            return chunks
        except Exception:
            pass

    def check_chunk_size(self, chunk, datapath) -> Iterator[Chunk]:
        """Verify chunk size and apply fallback TokenChunker if oversized.

        Args:
            chunk: Input chunk to check.
            datapath: Source file path.

        Yields:
            Chunk: Original chunk or sub-chunks if fallback was required.
        """

        if chunk.token_count > self.chunk_size:
            new = self.tokenchunker(chunk.text)
            yield from new
        else:
            yield chunk

    def metadata_add(self, chunked_file, data_path) -> list[dict[str, Any]]:
        """Enrich chunks with file metadata and convert to dictionary format.

        Args:
            chunked_file: Chonkie pipeline output object containing chunks.
            data_path: Path of the processed file.

        Returns:
            List of processed chunk dictionaries.
        """
        chunk_list: list[dict[str, Any]] = []
        for chunk in chunked_file.chunks:
            # print(chunk.token_count)
            if chunk:
                for sub_chunk in self.check_chunk_size(chunk, str(data_path)):
                    sub_chunk.metadata["file_path"] = str(data_path)
                    dict_chunk = sub_chunk.to_dict()
                    chunk_list.append(dict_chunk)
        return chunk_list

    def chunk_merger(self, chunked_file, min_size: int, max_size: int):
        print(len(chunked_file.chunks))
        new_chunk_list = []

        # Initialisation HORS de la boucle pour permettre l'accumulation
        start_index = 99999999999999999
        end_index = 0
        token_count = 0
        new_str = ""
        current_metadata = {}

        for chunk in chunked_file.chunks:
            if chunk.token_count >= min_size:
                # 1. Si on a des petits chunks en attente de fusion, on les
                # sauvegarde d'abord
                if token_count > 0:
                    new_chunk = Chunk(
                        text=new_str,
                        start_index=start_index,
                        end_index=end_index,
                        token_count=token_count,
                        metadata=current_metadata.copy()
                    )
                    new_chunk_list.append(new_chunk)

                    # On remet à zéro pour les prochains petits chunks
                    start_index = 99999999999999999
                    end_index = 0
                    token_count = 0
                    new_str = ""
                    current_metadata = {}

                # 2. On ajoute le gros chunk tel quel
                new_chunk_list.append(chunk)

            elif chunk.token_count < min_size:
                # 1. Si l'ajout de CE petit chunk dépasse le max_size, on
                # sauvegarde ce qu'on a déjà
                if token_count > 0 and (
                        token_count + chunk.token_count) > max_size:
                    new_chunk = Chunk(
                        text=new_str,
                        start_index=start_index,
                        end_index=end_index,
                        token_count=token_count,
                        metadata=current_metadata.copy()
                    )
                    new_chunk_list.append(new_chunk)

                    # On remet à zéro
                    start_index = 99999999999999999
                    end_index = 0
                    token_count = 0
                    new_str = ""
                    current_metadata = {}

                # 2. On accumule le petit chunk actuel
                start_index = chunk.start_index if chunk.start_index < start_index else start_index
                end_index = chunk.end_index

                # Gestion du séparateur de texte
                if new_str == "":
                    new_str = chunk.text
                else:
                    new_str += f"\n\n {chunk.text}"

                token_count += chunk.token_count

                if hasattr(chunk, 'metadata') and chunk.metadata:
                    current_metadata.update(chunk.metadata)

        # À la fin de la boucle, il faut sauvegarder les derniers petits chunks
        # s'il en reste
        if token_count > 0:
            new_chunk = Chunk(
                text=new_str,
                start_index=start_index,
                end_index=end_index,
                token_count=token_count,
                metadata=current_metadata.copy()
            )
            new_chunk_list.append(new_chunk)

        # On remplace l'ancienne liste par la nouvelle et on retourne l'objet
        chunked_file.chunks = new_chunk_list
        return chunked_file


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
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    pipeline = ChunkingPipeline()
    pipeline.run()
