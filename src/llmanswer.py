from typing import Any
from ollama import chat
from pathlib import Path
from tqdm import tqdm
from src.models import (MinimalAnswer,
                        MinimalSource,
                        StudentSearchResults,
                        StudentSearchResultsAndAnswer,
                        )


class MessagePrep:
    """Format prompt messages with retrieved context
      for dataset question evaluation."""

    def __init__(self, path: Path) -> None:
        """Initialize the path to precomputed search results."""
        self.retrieve = path

    def question_add(
            self, question_id: str) -> tuple[str, list[MinimalSource]]:
        """Build a context-injected prompt and
          return candidate sources for a question."""
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
    """Format prompt messages with retrieved context for a single query."""

    def __init__(self) -> None:
        """Initialize the path to solo search results."""
        self.retrieve = Path("data/output/search_results/"
                             "UnansweredQuestions/solo_answer.json")

    def question_add(
            self, question: str) -> str:
        """Build a context-injected prompt and
          return candidate sources for a single question."""
        context_text = ""
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
            f"{context_text[:12000]}\n"
            f"---------------------\n"
            f"Given the context information and no prior knowledge, "
            f"answer the question: {question_text}"
        )
        return message


class Answer:
    """Batch answer generator using Ollama for evaluation datasets."""

    def __init__(self,
                 student_search_results_path: str = (
                     "data/output/search_results/"
                     "UnansweredQuestions/dataset_code_public.json"),
                 k: int = 5, save_directory: str = "data/output/"
                 "search_results/UnansweredQuestions") -> None:
        """Initialize target model, dataset paths, and retrieval parameters."""
        self.model = 'qwen3:0.6b'
        self.question = Path(student_search_results_path)
        self.save_path = save_directory

    def __call__(self) -> Any:
        """Execute the batch question answering pipeline."""
        self.process_answers()

    def process_answers(self) -> None:
        """Iterate over dataset questions, query Ollama with context,
          and export answers to JSON."""
        total_q = 0
        content = MessagePrep(self.question)
        with open(self.question, "r", encoding="UTF-8") as f:
            question_data = StudentSearchResults.model_validate_json(f.read())
        llm_answer_list = []
        for q in question_data.search_results:
            total_q += 1

        for q in tqdm(question_data.search_results,
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

            # print(f"---{response.message.content}---")
            results.append(response.message.content)
            llm_answer_list.append(llm_answer)

            output = StudentSearchResultsAndAnswer(
                search_results=llm_answer_list, k=10)
            savefile = Path(f"{self.save_path}/llm_answer.json")
            savefile.parent.mkdir(parents=True, exist_ok=True)
            savefile.write_text(
                output.model_dump_json(indent=2), encoding="UTF-8")
        print("finish")


class SoloAnswer:
    """Single-query answer generator using Ollama."""

    def __init__(self, query: str, k: int) -> None:
        """Initialize the query, top-k parameter, and target model."""
        self.model = 'qwen3:0.6b'
        self.question = query
        self.k = k

    def __call__(self) -> Any:
        """Execute the single-query answer pipeline."""
        self.answer()

    def answer(self) -> None:
        """Query Ollama with context retrieved for the single question and
          print the output."""
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
        print("\n---LLM ANSWER---")
        print(response.message.content)
        print("---END---")
