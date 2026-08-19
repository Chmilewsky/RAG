from typing import Any
from ollama import chat
from pathlib import Path
from tqdm import tqdm
from src.models import (MinimalAnswer,
                        MinimalSource,
                        StudentSearchResults,
                        StudentSearchResultsAndAnswer,
                        RagDataset
                        )


class MessagePrep:
    def __init__(self) -> None:
        self.retrieve = Path("data/output/search_results/"
                             "UnansweredQuestions/dataset_code_public.json")

    def question_add(
            self, question_id: str) -> tuple[str, list[MinimalSource]]:
        context_text = ""
        sources: list[MinimalSource] = []
        with open(self.retrieve, "r", encoding="UTF-8") as f:
            retrieve_data = StudentSearchResults.model_validate_json(f.read())
        for q in retrieve_data.search_results:
            if q.question_id == question_id:
                sources = q.retrieved_sources
                question_text = q.question
                for r in q.retrieved_sources:
                    context_text += f"{r.chunk_txt} \n\n"
                break
        message = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"Given the context information and no prior knowledge, "
            f"answer the question: {question_text}"
        )
        message = message[:4000]
        return message, sources


class SoloMessagePrep:
    def __init__(self) -> None:
        self.retrieve = Path("data/output/search_results/"
                             "UnansweredQuestions/solo_answer.json")
    def question_add(
            self, question: str) -> tuple[str, list[MinimalSource]]:
        context_text = ""
        sources: list[MinimalSource] = []
        with open(self.retrieve, "r", encoding="UTF-8") as f:
            retrieve_data = StudentSearchResults.model_validate_json(f.read())
        msg = retrieve_data.search_results[0]
        question_text = msg.question
        q = retrieve_data.search_results[0]
        for r in q.retrieved_sources:
            context_text += f"{r.chunk_txt} \n"
        message = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"Given the context information and no prior knowledge, "
            f"answer the question: {question_text}"
        )

        message = message[:12000]
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
        total_q = 0
        content = MessagePrep()
        with open(self.question, "r", encoding="UTF-8") as f:
            question_data = RagDataset.model_validate_json(f.read())
        llm_answer_list = []
        output = []
        for q in question_data.rag_questions:
            total_q += 1

        for q in tqdm(question_data.rag_questions,
                      desc="Answer", total=total_q):
            results = []
            msg, source = content.question_add(q.question_id)
            messages = [
                {
                    'role': 'system',
                    'content': (
                        "You are a helpful and precise coding assistant. "
                        "Read the provided context"
                        " carefully and extract the relevant "
                        "information to answer the user's question."
                        " Be concise and factual."
                    )
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
        print("finish")


class SoloAnswer:
    def __init__(self, query, k) -> None:
        self.model = 'qwen3:0.6b'
        self.question = query
        self.k = k

    def __call__(self) -> Any:
        self.answer()

    def answer(self):
        content = SoloMessagePrep()
        msg = content.question_add(self.question)
        messages = [
            {
                'role': 'system',
                'content': (
                    "You are a helpful and precise coding assistant. "
                        "Read the provided context carefully"
                        " and extract the relevant "
                        "information to answer the user's question."
                        " Be concise and factual."
                )
            },
            {
                'role': 'user',
                'content': f"{msg}"
            }
        ]
        response = chat(model=self.model, messages=messages)
        print(f"---{response.message.content}---")
