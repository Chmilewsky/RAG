from typing import Any
from tqdm import tqdm
import json
import chromadb
from pathlib import Path
import shutil


class SemanticEmbeddings:
    """Manage local ChromaDB initialization and batch ingestion of text chunks."""

    def __init__(self) -> None:
        """Reset the local vector storage directory and initialize a persistent collection."""
        self.database_path = Path("data/intern_output/vector_DataBase")
        if self.database_path.exists():
            shutil.rmtree(self.database_path)
        self.client = chromadb.PersistentClient(
            path=self.database_path)
        self.collection = self.client.create_collection(name="my_collection")

    def __call__(self,) -> None:
        """Execute the database ingestion pipeline."""
        self.FillingDb()

    def FillingDb(self):
        """Read chunk data from JSONL and
          add documents with metadata to ChromaDB in batches."""
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
