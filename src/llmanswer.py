from ollama import chat
from pathlib import Path
import json

# response = chat(
#     model='qwen3:0.6b',
#     messages=[{'role': 'user', 'content': 'Hello!'}],
# )
# print(response.message.content)


def main():
    test = Answer(
        "data/datasets/UnansweredQuestions/dataset_code_public.json")
    test.open_file()


class Answer:
    def __init__(self, question_file) -> None:
        self.model = 'qwen3:0.6b'
        self.question = Path(question_file)

    def open_file(self):
        with open(self.question, "r", encoding="UTF-8") as f:
            question_data = json.load(f)
        results = []
        for q in question_data["rag_questions"]:
            messages = [
                {
                    'role': 'system',
                    'content': 'you are a RAG assistant.'
                },
                {
                    'role': 'user',
                    'content': f"Contexte:Question:\n{q['question']}"
                }
            ]
            response = chat(model=self.model, messages=messages, stream=True)
            for chunk in response:
                results.append(chunk.message.content)
                print(chunk.message.content, end='', flush=True)


if __name__ == "__main__":
    main()
