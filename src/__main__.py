import fire
from src.index import (
    ChunkIndex,
    SearchDataset,
    Search,
    SoloAnswer,
    AnswerDataset)


def main():
    # test = FileChunker()
    fire.Fire({
        "index": ChunkIndex,
        "search_dataset": SearchDataset,
        "search": Search,
        "answer": SoloAnswer,
        "answer_dataset": AnswerDataset
    })


if __name__ == "__main__":
    main()
