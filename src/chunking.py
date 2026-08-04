from chonkie import Pipeline
from magika import Magika
from pathlib import Path
import json
from typing import Any


class ChunkingPipeline:
    """Orchestrates scanning, chunking, and writing."""

    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./vllm-0.10.1",
                 output_path: str = "chunk_data.jsonl") -> None:
        self.scanner = FileScanner(dataset_path)
        self.chunker = FileChunker(chunk_size=chunk_size)
        self.writer = JsonWriter(output_path=output_path)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.run()

    def run(self):
        for file in self.scanner.folder_or_file():
            chunks = self.chunker.file_type_filter(file)
            self.writer.write(chunks)


class FileScanner:
    def __init__(self, dataset_path: str) -> None:
        self.data_path = dataset_path

    def folder_or_file(self) -> Any:
        p = Path(self.data_path)
        if not p.exists():
            print("no file")
        elif p.is_dir():
            print("folder")
            for item in p.rglob("*"):
                if item.is_file():
                    yield item
        elif p.is_file():
            print("file")
            yield p


class FileChunker:

    def __init__(self, chunk_size: int = 2000) -> None:
        self.chunk_size = chunk_size
        self.m = Magika()

    def file_type_filter(self, data):

        if data.suffix == ".py":
            chunks = self.py_chonking(data)
            return chunks
        elif data.suffix == ".md":
            chunks = self.md_chonking(data)
            return chunks
        elif data.suffix in [".yaml", ".cu", ".sh", ".toml"]:
            chunks = self.magika_chonking(data)
            return chunks

    def md_chonking(self, data) -> dict:
        data_path = str(data)
        # print(data_path)
        md_pipeline = (
            Pipeline().fetch_from(
                "file", path=data_path)
            .process_with("markdown")
            .chunk_with("recursive", chunk_size=self.chunk_size,
                        tokenizer="character", recipe="markdown").run())
        chunks = self.metadata_add(md_pipeline, data)
        return chunks

    def py_chonking(self, data) -> dict:
        data_path = str(data)
        # print(data_path)
        py_pipeline = (
            Pipeline().fetch_from(
                "file", path=data_path)

            .chunk_with("code", language="python", chunk_size=self.chunk_size,
                        tokenizer="character").run())
        chunks = self.metadata_add(py_pipeline, data)
        return chunks

    def magika_chonking(self, data) -> dict | None:
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

    def metadata_add(self, chunked_file, data_path):
        chunk_list = []
        for chunk in chunked_file.chunks:
            chunk.metadata["file_path"] = str(data_path)
            dict_chunk = chunk.to_dict()
            chunk_list.append(dict_chunk)

        return chunk_list


class JsonWriter:

    def __init__(self, output_path) -> None:
        self.output_path = output_path

    def write(self, chunks):
        with open(self.output_path, "a", encoding="UTF-8") as f:
            f.write(json.dumps(chunks, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    pipeline = ChunkingPipeline()
    pipeline.run()
