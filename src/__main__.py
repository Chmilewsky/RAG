import fire
from src.index import ChunkIndex, SearchDataset


def main():
    # test = FileChunker()
    fire.Fire({
        "index": ChunkIndex,
        "search_dataset": SearchDataset
    })


if __name__ == "__main__":
    main()
