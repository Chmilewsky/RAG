import bm25s
from pathlib import Path
import json


class IndexRetriever:
    def __init__(self, dataset_path, save_directory, k) -> None:
        self.dataset = Path(dataset_path)
        self.k = k
        self.save_path = Path(save_directory)

        self.import_retriever = bm25s.BM25.load("./data/processed")

    def __call__(self) -> None:
        self.retriever()

    def retriever(self):

        with open(self.dataset, "r", encoding="UTF-8") as f:
            question = json.load(f)
        for q in question["rag_questions"]:
            query = q["question"]
            query_tokens = bm25s.tokenize(query)

            results, scores = self.import_retriever.retrieve(
                query_tokens, k=self.k)
        #     # print(results)
        #     # print(scores)
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
        # print(f"Score: {score:.4f} | Document: {doc}")
        output_path = Path(
            "./data/output/search_results/UnansweredQuestions/index.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="UTF-8") as f:
            json.dump(final_json, f, indent=2)
