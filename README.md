*This project has been created as part of the 42 curriculum by gchmilew.*

# Description

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** pipeline.

Retrieval-Augmented Generation is a technique that enables a Large Language Model (LLM) to reference external knowledge sources—such as local documents, source code, and text files—to generate accurate, context-grounded answers and eliminate hallucinations.

### How the Pipeline Works

1. **Document Ingestion & Chunking**: Target files are loaded and divided into segments under a defined maximum character threshold while preserving semantic integrity (e.g., splitting with overlap for plain text, class/method boundaries for code, or header sections for Markdown). Each chunk is stored with its structural metadata, including source file provenance and start/end character offsets.
2. **Lexical Indexing (BM25)**: Chunks are tokenized (filtering stop words and applying stemming) and indexed using the BM25 algorithm. BM25 scores terms based on term frequency (TF) and inverse document frequency (IDF) while normalizing for document length.
3. **Query Matching & Retrieval**: When a query is submitted, it undergoes identical tokenization and scoring against the index to identify and rank the top-$k$ most relevant document chunks.
4. **Context Augmentation & Generation**: The top-$k$ chunks are formatted into an augmented context block and injected into the LLM prompt alongside the original query (`Prompt = Retrieved Context + User Question`), allowing the model to produce a grounded response.

---

# Instructions

The project provides both a native Command-Line Interface (CLI) built with **Python Fire** and a `Makefile` shortcut suite:

### Makefile Targets

| Command | Description |
| --- | --- |
| `make install` | Set up virtual environment and install all dependencies via `uv sync` |
| `make index` | Chunk source documents and build the BM25 lexical index |
| `make search` | Run a test query and display top-k matching sources in the terminal |
| `make search_dataset` | Execute batch retrieval across a benchmark query dataset |
| `make answer` | Generate an LLM response for a single query using retrieved context |
| `make answer_dataset` | Generate LLM answers for a dataset and export results to JSON |
| `make eval` | Compute retriever evaluation metrics (Recall@k) against reference baselines |
| `make semantic` | *(Bonus)* Chunk documents and build vector embeddings in ChromaDB |
| `make hybrid` | *(Bonus)* Run batch hybrid retrieval combining BM25 and ChromaDB via RRF |

### Python Fire CLI Commands

Every pipeline step can be invoked directly as specified by the evaluation protocol:

```bash
# Ingest corpus and build BM25 index
uv run python -m src index --max_chunk_size 2000 --dataset_path data/raw

# Retrieve top-k sources for a single query
uv run python -m src search "How to configure OpenAI server?" --k 5

# Batch retrieval over evaluation dataset (generates StudentSearchResults JSON)
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

# Single query answer generation with LLM
uv run python -m src answer "How to configure OpenAI server?" --k 5

# Batch answer generation (generates StudentSearchResultsAndAnswer JSON)
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Custom Recall@k evaluation against ground truth
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

# (Bonus) Build vector embeddings database with ChromaDB
uv run python -m src semantic --max_chunk_size 2000 --dataset_path data/raw

# (Bonus) Run batch hybrid retrieval combining BM25 and ChromaDB via RRF
uv run python -m src hybrid --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

---

# Resources

* [Chonkie Documentation](https://docs.chonkie.ai/common/welcome) — Lightweight chunking library for RAG applications.
* [Chonkie GitHub Repository](https://github.com/feyninc/chonkie) — Source code and implementation concepts.
* [Magika by Google](https://github.com/google/magika) — Deep learning-based file type and programming language detection.
* [BM25S PyPI Package](https://pypi.org/project/bm25s/0.1.5/) — Fast BM25 implementation in Python.
* [BM25S GitHub Repository](https://github.com/xhluca/bm25s) — Lexical search indexer using sparse matrices.
* [BM25 Algorithm Overview (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/) — Algorithmic explanation of Best Matching 25.
* [TQDM Progress Bar Guide (GeeksforGeeks)](https://www.geeksforgeeks.org/python/python-how-to-make-a-terminal-progress-bar-using-tqdm/) — Progress bar utilities for batch operations.
* [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/getting-started) — Vector database for lightweight CPU semantic embeddings.

### AI Usage Description
In accordance with 42 guidelines, artificial intelligence tools were consulted during the development of this project:
* **Library Comparison**: Evaluated different third-party libraries and tools to assess their features, performance, and relevance to the project requirements.
* **Chunking Methodologies**: Deepened the conceptual understanding of various document splitting strategies (recursive markdown, AST code chunking, and token-based fallback).
* **Splitting Rules**: Designed custom hierarchical rules for Markdown chunking.
* **Lexical Search (BM25s)**: Clarified the internal mechanics, scoring principles, and parameter behavior of the BM25s retrieval algorithm.
* **Hybrid Retrieval Fusion**: Guided the conceptualization and logic behind combining lexical BM25 results with semantic search into a unified ranking.
* **Code Ownership & Validation**: All suggested ideas, structures, and implementations were manually tested, adapted, and fully understood before integration into the codebase.

---

# Additional

## System architecture

The system follows a sequential, decoupled retrieval and generation architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                  Raw Documents (data/raw)                    │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
        Python Files (.py)             Markdown Files (.md)
               │                               │
               ▼                               ▼
    ┌──────────────────────┐       ┌──────────────────────┐
    │  AST Code Chunker    │       │  Recursive Markdown  │
    │ (Chonkie Code Parser)│       │(Custom Header Rules) │
    └──────────┬───────────┘       └───────────┬──────────┘
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │    Normalized Chunks (JSONL)  │
               │  [file_path, start, end, text]│
               └───────┬───────────────┬───────┘
                       │               │
        ┌──────────────┴───────┐       └──────────────┬────────┐
        ▼                                             ▼        │
┌──────────────────────────────┐       ┌───────────────────────┐
│     Lexical Index (BM25s)    │       │   ChromaDB Vectors    │ (Bonus)
│  (Tokenization + BM25-k1/b)  │       │ (ONNX CPU Embeddings) │
└──────────────┬───────────────┘       └──────────────┬────────┘
               │                                      │
               │◄───────────── User Query ───────────►│
               │                                      │
               ▼                                      ▼
┌──────────────────────────────┐       ┌───────────────────────┐
│     Lexical Top-K Ranking    │       │ Semantic Top-K Ranking│
└──────────────┬───────────────┘       └──────────────┬────────┘
               │                                      │
               └───────────────┬──────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ Reciprocal Rank Fusion (RRF)  │ (Bonus Hybrid)
               │   Score = Σ 1 / (60 + Rank)   │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │  StudentSearchResults (JSON)  │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ Context Augmentation & Prompt │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │       LLM Engine (Ollama)     │
               │       (Qwen / Qwen3-0.6B)     │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ StudentSearchResultsAndAnswer │
               └───────────────────────────────┘
```

---

## Chunking strategy

Chunking is handled using the `chonkie` library with distinct strategies depending on the file type:

### 1. Chunking Methods

* **Code Chunking (`py_chonking` & `magika_chonking`)**: Applied to Python files (`.py`) and various source files. It splits along logical code boundaries (classes, functions, methods) to preserve code structure.
* **Recursive Chunking with Rules (`md_chonking`)**: Applied to Markdown files (`.md`). It uses custom hierarchical rules (`custom_markdown.json`) to split content based on markdown headers and section delimiters.
* **Simple Token Chunking (`brut_chunk`)**: Applied to plain text (`.txt`). It performs a direct cut at `max_chunk_size` with an automatic overlap (`chunk_size // 5`) to preserve continuity across chunk boundaries.


For diverse source and configuration files (`.yaml`, `.cu`, `.sh`, `.toml`, `.cpp`, `.json`, `.cuh`, `.jinja`, `.h`), the pipeline integrates Google's **Magika**:
* Rather than relying solely on file extensions, Magika inspects file content using a lightweight neural network to accurately identify the programming language.
* The detected label is passed directly to the code chunker (`language=lang`), enabling syntax-aware code chunking adapted to that specific language.

---

## Retrieval method

Indexing and searching are handled by `bm25s`, which breaks text into words, scores their importance, and ranks the results:

### What is BM25?

BM25 is a search algorithm. When you search for keywords, BM25 looks at all your documents and scores how relevant each one is to your question. It is an improved, smarter version of classic TF-IDF.

---

### Core Mechanics

* **Term Frequency (TF) with Diminishing Returns**: If a search word appears in a chunk, that chunk gets points. But repeating the word over and over gives less and less extra points. This prevents someone from "gaming" the search by repeating the same keyword 50 times.
* **Inverse Document Frequency (IDF)**: Rare words are much more informative than common words. A word that only appears in two documents (like *"optimizer"*) gives a huge score boost, while a word that appears everywhere (like *"file"*) gives very few points.
* **Document Length Fair Play**: Long documents naturally contain more words, so they would normally get more matches by sheer luck. BM25 penalizes unusually long chunks so that short, focused passages have a fair chance to win.

---

| Component | Mechanism | Objective |
| --- | --- | --- |
| **TF Saturation** | Caps the score boost when a word is repeated | Stops repetitive text from cheating |
| **IDF Weighting** | Checks how rare a word is across all documents | Gives the spotlight to unique, key words |
| **Length Normalization** | Compares chunk size against the average size | Keeps short, accurate snippets competitive |

---

### Parameter Tuning: `k1` and `b`

The indexer configures BM25 with specific parameters:

```python
retriever = bm25s.BM25(k1=1.4, b=0.75)
```

* **`k1 = 1.4` (Word Repetition Cap)**:
  * Controls how fast repeated words stop giving bonus points.
  * **Higher `k1`**: Repeating a word continues to increase the score.
  * **Lower `k1`**: Repeating a word flattens out quickly; saying it once or ten times gives almost the same points.
  * *Why 1.4?* It rewards chunks that genuinely use an important function name two or three times, without letting boilerplate repetition drown out rare terms.

* **`b = 0.75` (Length Penalty)**:
  * Controls how much we penalize chunks that are longer than the average document length.
  * **`b = 1.0`**: Full penalty (a chunk twice as long must have twice as many matches to get the same score).
  * **`b = 0.0`**: Zero penalty (long documents win easily).
  * *Why 0.75?* It stops massive documentation pages from winning just because they are long, allowing short, punchy code functions to rank high.

---

### Tokenization: Stemmer and Stop Words

Both the indexing pipeline and query retrieval apply identical tokenization via `bm25s`:

```python
stemmer = Stemmer.Stemmer("english")
tokens = bm25s.tokenize(text, stemmer=stemmer, stopwords="en")
```

* **Stemming (Snowball / PyStemmer)**:
  * **How we use it**: We create `stemmer = Stemmer.Stemmer("english")` and pass it to `bm25s.tokenize()`. It automatically trims suffixes from every word.
  * **How it works**: The Snowball algorithm strips word endings like *-ing*, *-ed*, *-tion*, or *-s*.
  * **Why it matters**: Words like *"configuring"*, *"configured"*, and *"configuration"* all boil down to their root *"configur"*. That way, if a user asks *"how to configure"*, it will easily find documents that talk about *"configuration"*.
* **Stop Words (`stopwords="en"`)**:
  * We automatically remove everyday filler words (*"the"*, *"is"*, *"at"*, *"which"*, *"for"*).
  * This keeps search scores focused only on meaningful keywords.

---

## Performance analysis

* **Chunk Size**: Larger chunks (close to the 2000-character limit) provide more context and keywords for BM25, making it easier to match relevant queries.
* **Chunk Overlap**: Overlapping chunks prevent keywords from being cut in half at boundaries, which directly improves Recall@k scores.
* **Code vs. Documentation**: Documentation queries typically achieve higher recall because explanations share natural vocabulary, while code queries depend heavily on exact identifier and function names.

---

## Design decisions

* **Why Chonkie**: Chosen for its simplicity compared to heavy frameworks like LangChain. In addition to being lightweight, Chonkie automatically calculates and provides character offsets (`start_index` and `end_index`) on each chunk, which is essential to satisfy the project's source location requirements.
* **Why BM25s**: A very fast, embedded lexical search library that runs locally with NumPy/SciPy. It avoids the overhead of managing a dedicated search server (like Elasticsearch) while saving indices directly to disk.

---

## Challenges faced

* **Markdown Pre-Processing Artifacts**: `chonkie`'s built-in Markdown splitter performed pre-extraction routines that aggressively stripped headers and fragmented content, degrading downstream chunk quality.
* **Solution**: Markdown files were ingested using standard text mode configured with explicit priority split rules, preserving document structure without unwanted pre-extraction.

---

## Bonus Part

In addition to the mandatory lexical pipeline, two bonuses are implemented:

### 1. Semantic Embeddings (ChromaDB Vector Store)
* While BM25 searches for exact words, semantic search looks for **meaning**. Even if you use different words, it understands what you mean.
* We store vectors locally in **ChromaDB** under `data/processed/vector_DataBase/`.
* It automatically runs the small **`all-MiniLM-L6-v2`** model on CPU using ONNX, matching the recommended model from the subject.
* Command: `uv run python -m src semantic --max_chunk_size 2000` (or `make semantic`).

### 2. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
* Hybrid search gives you the best of both worlds: BM25 catches exact variable and function names, while ChromaDB catches concepts and general meaning.
* **How RRF works simply**:
  * BM25 and vector search output completely different types of scores that cannot be easily compared or added together.
  * Instead of comparing raw score numbers, RRF looks only at the **rank position** (1st, 2nd, 3rd...) of each document in both result lists.
  * The higher a chunk appears in either list, the more points it earns. Chunks that rank well in both searches naturally climb to the very top of the final combined ranking.
* **How the scoring loop works in code**:
  ```python
  for rank, index in enumerate(bm25, start=1):
      id = self.chunks[index]["id"]
      rrf_scores[id] = rrf_scores.get(id, 0.0) + (1.0 / (k + rank))
  ```
  * `enumerate(..., start=1)` processes items in order of relevance, assigning rank 1 to the top match, 2 to the second, etc.
  * `1.0 / (k + rank)` awards points based on position: a higher rank (smaller number) earns a bigger fraction (e.g., rank 1 gets $1/(60+1) \approx 0.0164$, while rank 10 gets $1/(60+10) \approx 0.0143$). The smoothing constant `k = 60` prevents top ranks from overwhelmingly dominating.
  * `rrf_scores.get(id, 0.0) + ...` accumulates score: the exact same loop runs for both BM25 and ChromaDB. If a chunk appears in both lists, its scores are summed together, giving it an extra boost that catapults it to the top.
* Command: `uv run python -m src hybrid --dataset_path <path> --k 10 --save_directory <dir>` (or `make hybrid`).

---

## Example usage

The complete workflow can be executed using either the `Makefile` targets or direct `uv run` commands:

### End-to-End Pipeline (Makefile)

```bash
# 1. Install dependencies
make install

# 2. Chunk source documents and build the lexical BM25 index
make index

# 3. Retrieve top-k sources for all queries in a dataset
make search_dataset

# 4. Generate grounded answers using the Qwen3-0.6B LLM
make answer_dataset

# 5. Evaluate retrieval accuracy (Recall@k) against reference baselines
make eval
```

### End-to-End Pipeline (Native CLI)

```bash
# 1. Indexing with custom chunk size (max 2000 chars)
uv run python -m src index --max_chunk_size 2000 --dataset_path data/raw

# 2. Batch search on public evaluation dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Official moulinette evaluation
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 --max_context_length 2000

# 4. Generate LLM answers from search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Testing the Bonuses

```bash
# Build the ChromaDB semantic vector index
uv run python -m src semantic --max_chunk_size 2000

# Run hybrid BM25 + Vector retrieval fused with RRF
uv run python -m src hybrid \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```