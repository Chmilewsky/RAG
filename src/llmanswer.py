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
    test = Answer(
        "data/datasets/UnansweredQuestions/dataset_code_public2.json")
    test.open_file()


class MessagePrep:
    def __init__(self) -> None:
        self.chunk = Path("data/output/search_results/"
                          "UnansweredQuestions/dataset_code_public.json")

    def question_add(self, question_id: str) -> str:
        message = ""
        with open(self.chunk, "r", encoding="UTF-8") as f:
            question_data = json.load(f)
        for q in question_data["search_results"]:
            if q["question_id"] == question_id:
                message += f"question: {q['question']} \n"
                for r in q["retrieved_sources"]:
                    message += f"context {r['chunk_txt']} \n"
                break
        message = message[:800]
        return message


class Answer:
    def __init__(self, question_file) -> None:
        self.model = 'qwen3:0.6b'
        self.question = Path(question_file)

    def open_file(self):
        te = MessagePrep()

        with open(self.question, "r", encoding="UTF-8") as f:
            question_data = json.load(f)
        results = []

        # msg = te.question_add(question_data["rag_questions"][0]["question_id"])
        # print(msg)

        for q in question_data["rag_questions"]:
            msg = te.question_add(q["question_id"])
            # print(msg)
            messages = [
                {
                    'role': 'system',
                    'content': 'you are a RAG assistant.'
                },
                {
                    'role': 'user',
                    'content': f"{msg}"
                }
            ]
            response = chat(model=self.model, messages=messages)
            print(f"---{response.message.content}---")
            results.append(response.message.content)
            # for chunk in tqdm(response, desc="llm answer"):
            #     results.append(chunk.message.content)

        print("finish")


if __name__ == "__main__":
    main()
