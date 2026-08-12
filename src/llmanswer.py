from ollama import chat
from pathlib import Path
import json
from tqdm import tqdm

# response = chat(
#     model='qwen3:0.6b',
#     messages=[{'role': 'user', 'content': 'Hello!'}],
# )
# print(response.message.content)


def main():
    # test = Answer(
    #     "data/datasets/UnansweredQuestions/dataset_code_public.json")
    # test.open_file()
    te = MessagePrep(question="data/datasets/UnansweredQuestions/"
                     "dataset_code_public2.json",
                     search_data="data/output/search_results/"
                     "UnansweredQuestions/dataset_code_public.json")


class MessagePrep:
    def __init__(self, question, search_data) -> None:
        self.question = Path(question)
        self.chunk = Path(search_data)
        self.message = ""
        self.question_add()
        # print(self.message)

    def question_add(self) -> None:
        with open(self.question, "r", encoding="UTF-8") as f:
            question_data = json.load(f)
        for q in question_data["rag_questions"]:
            self.message += f"question: {q["question"]} \n"
            self.message += (f"{q["question_id"]} \n\n")
            print("--chunk id--")
            self.chunk_add(q["question_id"])
            print("\n")

    def chunk_add(self, question_id):
        chunk = Path(
            "data/output/search_results/UnansweredQuestions/"
            "dataset_code_public.json")
        print(question_id)
        with open(chunk, "r", encoding="UTF-8") as f:
            question_data = json.load(f)
        for answer in question_data["search_results"]:
            if answer["question_id"] == question_id:
                for d in answer["retrieved_sources"]:
                    print(d["file_path"])


# class Answer:
#     def __init__(self, question_file) -> None:
#         self.model = 'qwen3:0.6b'
#         self.question = Path(question_file)

#     def open_file(self):
#         with open(self.question, "r", encoding="UTF-8") as f:
#             question_data = json.load(f)
#         results = []
#         for q in question_data["rag_questions"]:
#             messages = [
#                 {
#                     'role': 'system',
#                     'content': 'you are a RAG assistant.'
#                 },
#                 {
#                     'role': 'user',
#                     'content': f"Contexte:Question:\n{q['question']}"
#                 }
#             ]
#             response = chat(model=self.model, messages=messages, stream=True)
#             for chunk in tqdm(response, desc="llm answer"):
#                 results.append(chunk.message.content)
#                 # print(chunk.message.content, end='', flush=True)
#         print(results)


if __name__ == "__main__":
    main()
