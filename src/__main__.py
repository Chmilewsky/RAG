from chonkie import Pipeline
import bm25s
import json


def main():
    # md_pipeline = (
    #     Pipeline().fetch_from(
    #         "file", dir="./vllm-0.10.1", ext=[".md"])
    #     .process_with("markdown")
    #     .chunk_with("recursive", chunk_size=2000,
    #                 tokenizer="character", recipe="markdown")
    #     .export_with("json",
    #                  file="chunks.jsonl",
    #                  lines=False).run())
    corpus = []
    chunks = []

    with open("chunk.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["text"])
            chunks.append(data)

    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    for i in retriever.scores:
        print(i)

    query = "a cat and a dog"
    query_tokens = bm25s.tokenize(query)

    results, scores = retriever.retrieve(query_tokens, k=2)

    for doc, score in zip(results[0], scores[0]):
        print(f"Score: {score:.4f} | Document: {doc}")

    # md_pipelin2 = (
#     Pipeline().fetch_from(
#         "file", path="vllm-0.10.1/RELEASE.md")
#     .process_with("markdown")
#     .chunk_with("recursive", chunk_size=2000,
#                 tokenizer="character")
#     .export_with("json",
#                  file="chunkssolo.jsonl",
#                  lines=False).run())
# md_pipelin3 = (
#     Pipeline().fetch_from(
#         "file", path="vllm-0.10.1/RELEASE.md")
#     .process_with("markdown")
#     .chunk_with("recursive", chunk_size=2000,
#                 tokenizer="character", recipe="markdown")
#     .export_with("json",
#                  file="chunkssolorecipe.jsonl",
#                  lines=False).run())


if __name__ == "__main__":
    main()
