*This project has been created as part of the 42 curriculum by gchmilew.*

# Description

This project implements a lightweight **Retrieval-Augmented Generation (RAG)** pipeline.

Retrieval-Augmented Generation is a technique that enables a Large Language Model (LLM) to reference external knowledge sources—such as local documents, source code, and text files—to generate accurate, context-grounded answers and eliminate hallucinations.

### How the Pipeline Works

1. **Document Ingestion & Chunking**: Target files are loaded and divided into segments under a defined maximum character threshold while preserving semantic integrity (e.g., splitting at chapter breaks for plain text, class/method boundaries for code, or header sections for Markdown). Each chunk is stored with its structural metadata, including source file provenance and start/end character offsets.
2. **Lexical Indexing (BM25)**: Chunks are indexed using the BM25 algorithm. BM25 scores terms based on term frequency (TF) and inverse document frequency (IDF) while adjusting for document length and penalizing non-discriminative stop words (e.g., *a*, *is*, *it*, *for*).
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
* [BM25S PyPI Package](https://pypi.org/project/bm25s/0.1.5/) — Fast BM25 implementation in Python.
* [BM25S GitHub Repository](https://github.com/xhluca/bm25s) — Lexical search indexer using sparse matrices.
* [BM25 Algorithm Overview (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/) — Algorithmic explanation of Best Matching 25.
* [Python AST Module (W3Schools)](https://www.w3schools.com/python/ref_module_ast.asp) — Reference for Abstract Syntax Tree parsing in Python.
* [TQDM Progress Bar Guide (GeeksforGeeks)](https://www.geeksforgeeks.org/python/python-how-to-make-a-terminal-progress-bar-using-tqdm/) — Progress bar utilities for batch operations.
* [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/getting-started) — Vector database for lightweight CPU semantic embeddings.

### AI Usage Description
In accordance with 42 guidelines, artificial intelligence tools were consulted during the development of this project:
* **Library Comparison**: Evaluated different third-party libraries and tools to assess their features, performance, and relevance to the project requirements.
* **Chunking Methodologies**: Deepened the conceptual understanding of various document splitting strategies (recursive markdown, AST code chunking, and token-based fallback).
* **Lexical Search (BM25s)**: Clarified the internal mechanics, scoring principles, and parameter behavior of the BM25s retrieval algorithm.
* **Database & Index Management**: Assisted in designing the data storage pipeline, metadata tracking, and structured persistence of chunked data.
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

Chunking is handled using the `chonkie` library:

* **Format-Specific Splitting**: Provides tailored splitting rules based on file types to maintain semantic cohesion.
* **Priority-Based Hierarchy**: Splitting rules prioritize structural breakpoints (e.g., function boundaries, classes, markdown headers, paragraph breaks) rather than cutting mid-sentence.
* **Boundary Validation & Sliding Fallback**: Semantic splitting can occasionally exceed maximum character limits when avoiding awkward breaks. To enforce strict token limits, chunks undergo size validation. Chunks that exceed the boundary are processed through a fixed-size fallback splitter with character overlap to preserve context across splits.

---

## Retrieval method

Retrieval and indexing are powered by `bm25s`, which handles tokenization, corpus scoring, and ranking:

### What is BM25?

BM25 is a ranking algorithm used by retrieval engines (e.g., Elasticsearch, Lucene) to score document relevance against a search query, improving upon standard TF-IDF.

---

### Core Mechanics

* **Term Frequency (TF) Saturation**: Tracks query term occurrences with diminishing returns. Beyond a saturation threshold, repeated occurrences no longer scale the score linearly, mitigating keyword stuffing.
* **Inverse Document Frequency (IDF)**: Dynamically weights term specificity. Common words across the corpus (e.g., *"the"*, *"using"*) receive minimal weight, while rare terms receive higher relevance scores.
* **Document Length Normalization**: Adjusts scoring relative to average corpus length, penalizing artificially inflated match counts in long documents to keep evaluation fair for concise texts.

---

| Component | Mechanism | Objective |
| --- | --- | --- |
| **TF Saturation** | Applies a diminishing return ceiling to term counts | Prevents keyword spam bias |
| **IDF Penalty** | Evaluates document frequency across the whole corpus | Prioritizes rare, discriminative terms |
| **Length Normalization** | Compares document size against corpus average | Eliminates document length bias |

---

### Hyperparameter Tuning: `k1` and `b`

The indexer configures BM25 with specific parameters:

```python
retriever = bm25s.BM25(k1=1.4, b=0.75)
```


* **`k1 = 1.4` (Term Frequency Saturation)**:
  * Controls the non-linear saturation rate of term frequency.
  * **Higher `k1`**: The score scales more with repeated occurrences of a query word.
  * **Lower `k1`**: The score saturates rapidly, making repeated occurrences contribute little beyond the initial appearance.
  * *Impact*: With `k1 = 1.4`, snippets that legitimately reference a class or function name multiple times receive higher relevance without allowing boilerplate repetitions to overpower rare keywords.

* **`b = 0.75` (Document Length Normalization)**:
  * Regulates the penalty applied to chunks longer than the corpus average length ($\text{avgdl}$).
  * **`b = 1.0`**: Complete length penalty (a chunk twice as long needs twice as many keyword matches).
  * **`b = 0.0`**: Disables length normalization entirely, giving an unfair advantage to larger chunks.
  * *Impact*: With `b = 0.75`, long documentation pages are prevented from dominating the ranking simply because they contain more tokens, while concise and highly targeted code snippets remain competitive.

---



## Performance analysis

Empirical evaluation reveals distinct performance variations based on the chunking strategy:

* **Fixed Overlap vs. Semantic Splits**: Brute-force fixed chunking with overlap consistently scored higher on **Recall@k** benchmarks than clean semantic chunking (at functions, titles, or chapters).
* **Metric Bias**: Standard Recall@k measures character and token index overlap against reference answer spans. Uniformly distributed, overlapping chunks offer broader spatial coverage across document offsets, increasing the probability of intersecting ground-truth spans compared to variable-sized semantic segments.

---

## Design decisions

* **Lightweight Architecture over Monolithic Frameworks**: `chonkie` was chosen instead of heavy frameworks such as LangChain. It provides a focused, dependency-light open-source implementation with direct control over chunk boundaries and character offsets.
* **Embedded Lexical Search**: Using `bm25s` eliminates the need for dedicated search infrastructure (e.g., Elasticsearch, OpenSearch) while delivering high-throughput lexical retrieval and native sparse matrix persistence.

---

## Challenges faced

* **Markdown Pre-Processing Artifacts**: `chonkie`'s built-in Markdown splitter performed pre-extraction routines that aggressively stripped headers and fragmented content, degrading downstream chunk quality.
* **Solution**: Markdown files were ingested using standard text mode configured with explicit priority split rules, preserving document structure without unwanted pre-extraction.


* **Strict Boundary Enforcement**: Balancing semantic unit completeness with hard character limits required implementing a secondary validation pass with overlapping cuts.

---

## Bonus Part

In addition to the mandatory lexical pipeline, two bonuses are implemented:

### 1. Semantic Embeddings (ChromaDB Vector Store)
* Semantic vector processing and storage are handled entirely by **ChromaDB**.
* By default, ChromaDB's `DefaultEmbeddingFunction` automatically runs the **`all-MiniLM-L6-v2`** model using **ONNX Runtime (100% CPU)**, matching the exact lightweight model recommended in the subject specification.
* The vector collection is persisted locally on disk under `data/processed/vector_DataBase/`.
* Command: `uv run python -m src semantic --max_chunk_size 2000` (or `make semantic`).

### 2. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
* Combines exact keyword search (BM25) and semantic meaning search (ChromaDB) into a single result list.
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