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

The project includes a `Makefile` with targets to manage the complete lifecycle from dependency setup to evaluation:

| Command | Description |
| --- | --- |
| `make install` | Set up the virtual environment and install all dependencies |
| `make index` | Chunk source documents and build the BM25 lexical index |
| `make search` | Run a test query and display the top ranked chunks with relevance scores |
| `make search_dataset` | Execute retrieval across the entire benchmark query dataset |
| `make answer` | Generate an LLM response for a single query using retrieved context |
| `make answer_dataset` | Run the complete question dataset through the LLM and export answers to JSON |
| `make evaluate` | Compute retriever evaluation metrics (e.g., Recall@k) against reference baselines |

---

# Resources

* [Chonkie Documentation](https://docs.chonkie.ai/common/welcome) — Lightweight chunking library for RAG applications.
* [Chonkie GitHub Repository](https://github.com/feyninc/chonkie) — Source code and implementation concepts.
* [BM25S PyPI Package](https://pypi.org/project/bm25s/0.1.5/) — Fast BM25 implementation in Python.
* [BM25S GitHub Repository](https://github.com/xhluca/bm25s) — Lexical search indexer using sparse matrices.
* [BM25 Algorithm Overview (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/) — Algorithmic explanation of Best Matching 25.
* [Python AST Module (W3Schools)](https://www.w3schools.com/python/ref_module_ast.asp) — Reference for Abstract Syntax Tree parsing in Python.
* [TQDM Progress Bar Guide (GeeksforGeeks)](https://www.geeksforgeeks.org/python/python-how-to-make-a-terminal-progress-bar-using-tqdm/) — Progress bar utilities for batch operations.
* https://docs.trychroma.com/docs/overview/getting-started

---

# Additional

## System architecture

The system follows a sequential, decoupled retrieval and generation architecture:

```
┌─────────────────┐
│ Raw Documents   │ (Text, Code, Markdown)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chunking Engine │ (Chonkie: structural boundaries + metadata extraction)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Lexical Indexer │ (bm25s: tokenization, BM25 scoring, sparse matrices)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Retriever │ (Token matching, top-k scoring)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prompt Assembly │ (System instructions + Retrieved Context + Query)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Engine      │ (Grounded answer generation -> JSON output)
└─────────────────┘

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

## Example usage

To run the entire pipeline from environment setup to final response generation:

```bash
# 1. Install dependencies
make install

# 2. Chunk source documents and build the BM25 index
make index

# 3. Retrieve relevant sources for all dataset queries
make search_dataset

# 4. Generate grounded LLM answers for all dataset queries
make answer_dataset

# 5. Evaluate retriever accuracy against reference baselines
make evaluate

```