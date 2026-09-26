# RAGForge

A retrieval-augmented generation (RAG) system for question-answering over a corpus of research papers — built and evaluated as a testbed for measuring **retrieval quality** and **grounded answer behavior**, not just "does the LLM sound plausible."

RAGForge ingests PDFs, chunks and embeds them into a local vector store, retrieves the most relevant passages for a question, and generates an answer that is required to cite its sources or explicitly refuse when the answer isn't in the corpus. It ships with a 50-question evaluation benchmark to quantitatively track retrieval and refusal quality as the pipeline changes.

## ✨ Features

- **PDF ingestion pipeline** — chunks research papers with `RecursiveCharacterTextSplitter` and embeds them locally using `sentence-transformers/all-mpnet-base-v2` (no external embedding API required)
- **Persistent vector store** — ChromaDB-backed collection with per-source ingestion diagnostics (chunk counts, chunk-length stats, near-empty chunk detection)
- **Grounded retrieval + generation** — retrieves top-k chunks via similarity search and generates answers with a local LLM (Ollama / `llama3.2:3b`), constrained to answer only from retrieved context and cite `(source, page)`
- **Refusal-aware generation** — the model is explicitly instructed to refuse when a question falls outside the retrieved context, rather than hallucinating an answer
- **50-question evaluation benchmark** — measures Paper Hit@5, Page Hit@5, MRR, refusal rate, and false-refusal rate across 6 research papers
- **REST API + web UI** — a FastAPI backend exposing the pipeline over HTTP, with a lightweight HTML/JS frontend for asking questions and inspecting retrieved chunks

## 🏗️ Architecture

```text
   PDFs (papers/)
        │
        ▼
  ┌─────────────┐      ┌──────────────────────┐
  │  ingest.py  │─────▶│   ChromaDB (persist)  │
  └─────────────┘      │  Research_Papers coll.│
                        └───────────┬──────────┘
                                    │ similarity_search_with_score
                                    ▼
                        ┌──────────────────────┐
                        │      query.py         │
                        │  retrieve → prompt →  │
                        │   Ollama (llama3.2:3b)│
                        └───────────┬──────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ▼                             ▼
             ┌───────────────┐            ┌────────────────┐
             │    api.py     │            │    eval.py      │
             │ FastAPI (REST)│            │ 50-Q benchmark  │
             └───────┬───────┘            │ eval_results.json│
                     │                    └────────────────┘
                     ▼
          ┌───────────────────────┐
          │  frontend/index.html   │
          │  (vanilla HTML/JS UI)  │
          └───────────────────────┘
```

## 🧰 Tech Stack

`Python` · `LangChain` · `ChromaDB` · `sentence-transformers` (HuggingFace embeddings) · `Ollama` (local LLM inference) · `FastAPI` · `uvicorn` · vanilla `HTML/CSS/JS`

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed locally, with the model pulled:
  ```bash
  ollama pull llama3.2:3b
  ```

### Setup

```bash
git clone https://github.com/2602-AyushG/RAGForge.git
cd RAGForge
pip install -r requirements.txt
```

Add your PDFs to `papers/` (see `data/README.md` for corpus details).

### 1. Ingest the corpus

```bash
python ingest.py
```
Chunks and embeds every PDF in `papers/` into a persistent Chroma vector store, printing per-source chunk stats along the way.

### 2. Ask a question from the CLI

```bash
python query.py "What is the main contribution of DenseNet?"
```

### 3. Run it as an API + web UI

```bash
uvicorn api:app --reload
```
Then open **http://localhost:8000** — enter a question and see the generated answer alongside the retrieved chunks (source, page, similarity score).

## 📊 Evaluation

The RAG system is evaluated using a 50-question benchmark grounded in the six research papers stored in the vector database.

### Evaluation Dataset

The benchmark contains:

* **46 in-corpus questions** based on the six research papers
* **4 out-of-scope questions** that should be refused
* Questions covering:
  * Direct factual retrieval
  * Paraphrased questions
  * Cross-paper questions
  * Out-of-scope but potentially confusing questions

Each in-corpus question is associated with an expected source paper and expected page range.

### Evaluation Metrics

The evaluation reports the following metrics:

* **Paper Hit@5** — whether the expected paper appears among the top 5 retrieved chunks
* **Page Hit@5** — whether a chunk from the expected page range appears among the top 5 retrieved chunks
* **MRR (Mean Reciprocal Rank)** — measures how highly the expected source appears in the retrieval results
* **Refusal Rate** — percentage of out-of-scope questions for which the model correctly refuses to answer
* **False-Refusal Rate** — percentage of valid in-corpus questions incorrectly refused by the model

The evaluation also compares selected questions with and without retrieved context to examine the effect of RAG on the generated answers.

### Latest Results (k=5, model: `llama3.2:3b`)

| Metric | Score |
|---|---|
| Paper Hit@5 | **95.7%** |
| Page Hit@5 | 37.0% |
| MRR | 0.908 |
| Refusal Rate (out-of-scope) | **100%** |
| False-Refusal Rate | **0%** |

Retrieval reliably finds the *correct paper* and the system never hallucinates on out-of-scope questions with zero false refusals — the current weak point is *page-level* precision (37%), which is the active area of investigation (chunking strategy across page boundaries, hybrid/rerank retrieval).

### Running the Evaluation

After the vector database has been created, run:

```bash
python eval.py
```

The evaluation prints retrieval results, generated answers, refusal statistics, and aggregate metrics to the terminal. A detailed machine-readable copy is also saved to `eval_results.json`.

### Current Evaluation Scope

The benchmark currently evaluates retrieval and answer generation across:

1. DenseNet
2. EfficientNet
3. Grad-CAM
4. Integrated Gradients
5. LIME
6. The project's comparative XAI study

The evaluation is intended to measure both **retrieval quality** and **grounded answer behavior**, rather than relying only on whether the language model produces a plausible answer.

## 📁 Project Structure

```text
RAGForge/
├── ingest.py            # PDF → chunks → embeddings → ChromaDB
├── query.py             # retrieval + grounded generation (CLI)
├── api.py               # FastAPI wrapper around query.py
├── frontend/
│   └── index.html       # minimal web UI for the API
├── eval.py               # 50-question evaluation benchmark
├── eval_results.json     # latest evaluation run (metrics + per-question detail)
├── data/
│   └── README.md         # corpus description (papers excluded via .gitignore)
├── scratch/               # exploratory Chroma/semantic-search scripts
└── requirements.txt
```

## 🗺️ Roadmap

- [ ] Hybrid retrieval (BM25 + dense) / reranking to improve Page Hit@5
- [ ] Centralized config (`config.py` / `.env`) instead of duplicated constants
- [ ] Automated tests (`pytest`) for ingestion and retrieval
- [ ] CI pipeline (lint + smoke tests on push)
