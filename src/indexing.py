from typing import Any
from pathlib import Path

import bm25s
import json


class Indexing:
    def __init__(self, k=10) -> None:
        self.corpus = []
        self.chunks = []
        self.question_str = []
        self.question_object = []
        self.k = k

    def __call__(self) -> Any:
        self.build_index()

    def build_index(self) -> None:
        list_dict_index = []
        try:

            with open("chunk_data.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.corpus.append(data["text"])
                    self.chunks.append(data)
        except Exception:
            print("no chunk_data.jsonl")

        corpus_tokens = bm25s.tokenize(self.corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        doc_question_path = "./datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
        code_question_path = "./datasets_public/public/UnansweredQuestions/dataset_code_public.json"

        with open(code_question_path, "r", encoding="UTF-8") as f:
            question = json.load(f)
        question_list_size = len(question["rag_questions"])
        for i in range(question_list_size):
            # print(question["rag_questions"][i]["question"])
            # self.question_str.append(question["question"])
            # self.question_object.append(question)

            query = question["rag_questions"][i]["question"]
            query_tokens = bm25s.tokenize(query)

            results, scores = retriever.retrieve(query_tokens, k=self.k)
            # print(results)
            # print(scores)

            for index in results[0]:
                print(f"{index}")
                dict_json = {"text": self.chunks[index]["text"],
                             "question_id": question["rag_questions"][i]["question_id"],
                             "question": question["rag_questions"][i]["question"],
                             "answer": "xx",
                             "sources": [
                    {
                        "file_path": self.chunks[index]["metadata"]["file_path"],
                        "first_character_index": self.chunks[index]["start_index"],
                        "last_character_index": self.chunks[index]["end_index"]
                    }
                ],
                }
                list_dict_index.append(dict_json)
                # print(f"Score: {score:.4f} | Document: {doc}")
            output_path = Path("./data/processed/index.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a", encoding="UTF-8") as f:
                for i in list_dict_index:
                    try:
                        f.write(json.dumps(i) + "\n")
                    except Exception:
                        print("ca merde")
