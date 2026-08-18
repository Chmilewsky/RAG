from typing import Any

from ollama import chat
from pathlib import Path
from tqdm import tqdm
from pydantic import ValidationError
from src.models import (MinimalAnswer,
                        MinimalSource,
                        StudentSearchResults,
                        StudentSearchResultsAndAnswer,
                        RagDataset
                        )

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

    def question_add(
            self, question_id: str) -> tuple[str, list[MinimalSource]]:
        message = ""
        sources: list[MinimalSource] = []
        with open(self.chunk, "r", encoding="UTF-8") as f:
            question_data = StudentSearchResults.model_validate_json(f.read())
        for q in question_data.search_results:
            if q.question_id == question_id:
                sources = q.retrieved_sources
                message += f"question: {q.question} \n"
                for r in q.retrieved_sources:
                    message += f"context {r.chunk_txt} \n"
                break
        message = message[:4000]
        return message, sources


class Answer:
    def __init__(self,
                 student_search_results_path=(
                     "data/datasets/UnansweredQuestions/"
                     "dataset_docs_public.json"),
                 k=5, save_directory="data/output/"
                 "search_results/UnansweredQuestions") -> None:
        self.model = 'qwen3:0.6b'
        self.question = Path(student_search_results_path)

    def __call__(self) -> Any:
        self.open_file()

    def open_file(self):
        content = MessagePrep()
        with open(self.question, "r", encoding="UTF-8") as f:
            question_data = RagDataset.model_validate_json(f.read())
        llm_answer_list = []
        output = []

        for q in question_data.rag_questions:
            results = []
            msg, source = content.question_add(q.question_id)
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
            llm_answer = MinimalAnswer(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=source,
                answer=response.message.content or "")

            print(f"---{response.message.content}---")
            results.append(response.message.content)
            llm_answer_list.append(llm_answer)

        output = StudentSearchResultsAndAnswer(
            search_results=llm_answer_list, k=10)
        savefile = Path("testouput.json")
        savefile.write_text(
            output.model_dump_json(indent=2), encoding="UTF-8")

        # for chunk in tqdm(response, desc="llm answer"):
        #     results.append(chunk.message.content)
        # pydantic pour sorti

        print("finish")


class SoloAnswer:
    def __init__(self, query, k) -> None:
        self.model = 'qwen3:0.6b'
        self.question = query
        self.k = k

    def __call__(self) -> Any:
        self.answer()

    def answer(self):
        content = MessagePrep()
        msg = content.question_add(self.question)
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


if __name__ == "__main__":
    main()
