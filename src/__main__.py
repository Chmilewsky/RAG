import fire
from src.index import ChunkIndex, SearchDataset, Search


def main():
    # test = FileChunker()
    fire.Fire({
        "index": ChunkIndex,
        "search_dataset": SearchDataset,
        "search": Search
    })


if __name__ == "__main__":
    main()
