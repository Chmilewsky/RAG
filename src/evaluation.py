from src.models import (StudentSearchResults, RagDataset,
                        AnsweredQuestion, UnansweredQuestion)
from typing import Any


class Eval:
    """Evaluate retrieval performance against source references."""

    def __init__(self, student_search_results_path: str =
                 "data/output/search_results"
                 "/UnansweredQuestions/dataset_code_public.json",
                 dataset_path: str =
                 "data/datasets/"
                 "AnsweredQuestions/dataset_code_public.json") -> None:
        """Load student search results and
          ground-truth dataset from JSON files."""

        with open(student_search_results_path, "r", encoding="Utf-8") as f:
            self.student_data = StudentSearchResults.model_validate_json(
                f.read())
        with open(dataset_path, "r", encoding="UTF-8") as g:
            self.answer = RagDataset.model_validate_json(g.read())

    def __call__(self) -> Any:
        """Execute the recall evaluation pipeline."""
        self.recall()

    def recall(self) -> None:
        """Compute and display Recall@1, Recall@3, Recall@5, and
          Recall@10 metrics."""
        total_question = 0
        good_answer1 = 0
        good_answer3 = 0
        good_answer5 = 0
        good_answer10 = 0

        for q in self.answer.rag_questions:
            total_question += 1
            for student in self.student_data.search_results:
                if student.question_id == q.question_id:
                    good_answer1 += self.krecall(
                        student.retrieved_sources, q, 1)
                    good_answer3 += self.krecall(
                        student.retrieved_sources, q, 3)
                    good_answer5 += self.krecall(
                        student.retrieved_sources, q, 5)
                    good_answer10 += self.krecall(
                        student.retrieved_sources, q, 10)

        rk1 = good_answer1 / total_question
        rk3 = good_answer3 / total_question
        rk5 = good_answer5 / total_question
        rk10 = good_answer10 / total_question

        print(f"recall@1: {rk1:.2f} ({rk1 * 100:.1f}%)")
        print(f"recall@3: {rk3:.2f} ({rk3 * 100:.1f}%)")
        print(f"recall@5: {rk5:.2f} ({rk5 * 100:.1f}%)")
        print(f"recall@10: {rk10:.2f} ({rk10 * 100:.1f}%)")

    def krecall(self, src: list, q: AnsweredQuestion
                | UnansweredQuestion, k: int) -> int:
        """Check whether at least one top-k retrieved source
          meets the file and overlap thresholds."""
        if isinstance(q, UnansweredQuestion) or not q.sources:
            return 0
        for i in range(k):
            if src[i].file_path == q.sources[0].file_path:
                x, y = (src[i].first_character_index,
                        src[i].last_character_index)
                a, b = (q.sources[0].first_character_index,
                        q.sources[0].last_character_index)
                if self.overlap((x, y), (a, b)) >= 0.05:
                    return 1

        return 0

    def overlap(self, src: tuple[int, int], answer: tuple[int, int]) -> float:
        """Compute the intersection-over-union ratio
          between two character index spans."""
        x, y = src
        a, b = answer
        intersection_start = max(x, a)
        intersection_end = min(y, b)
        intersection = max(0, intersection_end - intersection_start)
        if intersection == 0:
            return 0.0
        union_start = min(x, a)
        union_end = max(y, b)
        total_range = union_end - union_start
        if total_range <= 0:
            return 0.0
        result = intersection / total_range
        return result
