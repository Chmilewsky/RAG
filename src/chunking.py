from chonkie import Pipeline, CodeChunker
from magika import Magika
from pathlib import Path
import json


class FileChunker:
    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./vllm-0.10.1") -> None:
        self.data_path = dataset_path
        self.chunk_size = chunk_size
        self.m = Magika()
        self.output_path: Path = Path("chunk_data.jsonl")
        if self.output_path.exists():
            self.output_path.unlink()

    def __call__(self) -> None:
        """Default action"""
        self.folder_or_file()

    def folder_or_file(self) -> None:
        p = Path(self.data_path)
        if not p.exists():
            print("no file")
        elif p.is_dir():
            print("folder")
            self.folder_looping(p)
        elif p.is_file():
            print("file")
            self.file_type_filter(p)

    def folder_looping(self, data) -> None:
        for item in data.rglob("*"):
            self.file_type_filter(item)

    def file_type_filter(self, data) -> None:

        if data.suffix == ".py":
            self.py_chonking(data)
        elif data.suffix == ".md":
            self.md_chonking(data)
        elif data.suffix in [".yaml", ".cu", ".sh", ".toml"]:
            self.magika_chonking(data)

    def md_chonking(self, data) -> None:
        data_path = str(data)
        # print(data_path)
        md_pipeline = (
            Pipeline().fetch_from(
                "file", path=data_path)
            .process_with("markdown")
            .chunk_with("recursive", chunk_size=self.chunk_size,
                        tokenizer="character", recipe="markdown").run())
        with open(self.output_path, "a", encoding="UTF-8") as f:

            for chunk in md_pipeline.chunks:
                chunk.metadata["file_path"] = data_path
                dict_chunk = chunk.to_dict()
                f.write(json.dumps(dict_chunk, ensure_ascii=False) + "\n")

    def py_chonking(self, data) -> None:
        data_path = str(data)
        # print(data_path)
        py_pipeline = (
            Pipeline().fetch_from(
                "file", path=data_path)

            .chunk_with("code", language="python", chunk_size=self.chunk_size,
                        tokenizer="character").run())
        with open(self.output_path, "a", encoding="UTF-8") as f:
            for chunk in py_pipeline.chunks:
                chunk.metadata["file_path"] = data_path
                dict_chunk = chunk.to_dict()
                f.write(json.dumps(dict_chunk, ensure_ascii=False) + "\n")

    def magika_chonking(self, data) -> None:
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
            with open(self.output_path, "a", encoding="UTF-8") as f:
                for chunk in magika_pipeline.chunks:
                    chunk.metadata["file_path"] = data_path
                    dict_chunk = chunk.to_dict()
                    f.write(json.dumps(dict_chunk, ensure_ascii=False) + "\n")
        except Exception:
            pass
