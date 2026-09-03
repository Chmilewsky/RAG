from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """pydantic check for answer format"""
    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_txt: str | None = None


class UnansweredQuestion(BaseModel):
    """pydantic check for question format"""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """pydantic check for answer format"""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """pydantic check for question file"""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """pydantic check for answer format"""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """pydantic check for llm answer format"""
    answer: str


class StudentSearchResults(BaseModel):
    """pydantic check for llm answer format"""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """pydantic check for llm answer format"""
    search_results: list[MinimalAnswer]
    k: int
