from typing import Any
from tqdm import tqdm
import json

import chromadb


class SemanticEmbeddings:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path="data/intern_output")
        self.collection = self.client.create_collection(name="my_collection")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.FillingDb()

    def FillingDb(self):
        ids = []
        documents = []
        metadatas = []

        with open("./data/intern_output/chunk_data.jsonl",
                  "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="tokenizing"):
                data = json.loads(line)
                ids.append(data["id"])
                documents.append(data["text"])
                metadatas.append({
                    "filename": data['metadata']['filename'],
                    "filepath": data['metadata']['file_path']
                })
        batch_size = 500
        total = len(ids)
        for i in tqdm(range(0, total, batch_size),
                      desc="embedding and storing"):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size])
