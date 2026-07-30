import bm25s


class Indexing:
    corpus = []
    chunks = []

    with open("chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["text"])
            chunks.append(data)

    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    for i in retriever.scores:
        print(i)

    query = "server"
    query_tokens = bm25s.tokenize(query)

    results, scores = retriever.retrieve(query_tokens, k=3)

    for doc, score in zip(results[0], scores[0]):
        print(f"Score: {score:.4f} | Document: {doc}")
