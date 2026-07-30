from chonkie import Pipeline
import os


class FileChunker:
    def __init__(self, chunk_size: int = 2000,
                 dataset_path: str = "./vllm-0.10.1") -> None:
        self.data_path = dataset_path
        self.chunk_size = chunk_size

    def __call__(self) -> None:
        """Default action"""
        self.folder_or_file()

    def folder_or_file(self) -> None:
        if not os.path.exists(self.data_path):
            print("no file")
        elif os.path.isdir(self.data_path):
            print("folder")
            self.folder_looping(self.data_path)
        elif os.path.isfile(self.data_path):
            print("file")
            self.file_type_filter(self.data_path)

    def folder_looping(self, data):
        for root, dirs, files in os.walk(data):

            print(files)

    def file_type_filter(self, data):
        name, ext = os.path.splitext(data)
        print(ext)
        if ext == ".py":
            self.py_chonking(data)
        elif ext == ".md":
            self.md_chonking(data)
        elif ext == ".pdf":
            self.pdf_chonking(data)
        elif ext == ".txt":
            self.txt_chonking(data)

    def md_chonking(self, data):
        print(data)
        md_pipeline = (
            Pipeline().fetch_from(
                "file", path=data)
            .process_with("markdown")
            .chunk_with("recursive", chunk_size=self.chunk_size,
                        tokenizer="character", recipe="markdown").run())
        for chunk in md_pipeline.chunks:
            chunk.metadata["full_path"] = data
            print("\n----\n")
            print(f"{chunk.metadata}")
            print(chunk)

    def py_chonking(self, data):
        pass

    def pdf_chonking(self, data):
        pass

    def txt_chonking(self, data):
        pass
