import fire
from src.chunking import FileChunker


def main():
    # test = FileChunker()
    fire.Fire(FileChunker)


if __name__ == "__main__":
    main()
