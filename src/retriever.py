import bm25s
from pathlib import Path
import json
import Stemmer


class IndexRetriever:
    def __init__(self, dataset_path, save_directory, k) -> None:
        self.dataset = Path(dataset_path)
        self.k = k
        self.save_path = Path(save_directory)
        with open("data/intern_output/chunk_data.jsonl",
                  "r", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

        self.import_retriever = bm25s.BM25.load("./data/processed")

    def __call__(self) -> None:
        self.retriever()

    def retriever(self):
        list_dict_index = []
        stemmer = Stemmer.Stemmer("english")

        with open(self.dataset, "r", encoding="UTF-8") as f:
            question = json.load(f)
        for q in question["rag_questions"]:
            query = q["question"]
            query_tokens = bm25s.tokenize(query, stemmer=stemmer)

            results, scores = self.import_retriever.retrieve(
                query_tokens, k=self.k)
            retrieved_sources = []
            for index in results[0]:
                start = self.chunks[index]["start_index"]
                end = self.chunks[index]["end_index"]

                retrieved_sources.append(
                    {
                        "file_path": (self.chunks[index]
                                      ["metadata"]["file_path"]),
                        "first_character_index": start,
                        "last_character_index": end

                    })
                dict_json = {
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "retrieved_sources": retrieved_sources

                }
                list_dict_index.append(dict_json)
        final_json = {
            "search_results": list_dict_index,
            "k": self.k
        }

        savefile = self.save_path / self.dataset.name
        savefile.parent.mkdir(parents=True, exist_ok=True)
        print(savefile)
        try:
            with open(savefile, "w", encoding="UTF-8") as f:
                json.dump(final_json, f, indent=2)
        except Exception as e:
            print(e)
