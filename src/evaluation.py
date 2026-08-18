from src.models import StudentSearchResults, RagDataset
from typing import Any


class Eval:
    def __init__(self, student_search_results_path: str =
                 "data/output/search_results"
                 "/UnansweredQuestions/dataset_code_public.json",
                 dataset_path: str =
                 "data/datasets/AnsweredQuestions/dataset_code_public.json") -> None:
        with open(student_search_results_path, "r", encoding="Utf-8") as f:
            self.student_data = StudentSearchResults.model_validate_json(
                f.read())
        with open(dataset_path, "r", encoding="UTF-8") as g:
            self.answer = RagDataset.model_validate_json(g.read())

    def __call__(self) -> Any:
        self.recall()

    def recall(self):
        total_question = 0
        good_answer = 0
        for q in self.answer.rag_questions:
            total_question += 1
            for student in self.student_data.search_results:
                if student.question_id == q.question_id:
                    for src in student.retrieved_sources:
                        if src.file_path == q.sources[0].file_path:
                            x, y = (src.first_character_index,
                                    src.last_character_index)
                            a, b = (q.sources[0].first_character_index,
                                    q.sources[0].last_character_index)
                            if self.overlap((x, y), (a, b)) > 0.05:
                                good_answer += 1
        print(good_answer / total_question)

    def overlap(self, src: tuple, answer: tuple) -> float:
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
