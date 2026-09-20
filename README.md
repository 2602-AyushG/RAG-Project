# RAGForge

A local, end-to-end Retrieval-Augmented Generation (RAG) pipeline in Python that answers questions over a small collection of explainable-AI (XAI) research papers and cites the source file and page for every answer. It runs fully offline: local embeddings, a local vector store, and a local LLM via Ollama, so there are no API keys and no paid services.

## What it does

1. **Ingest** – parses PDFs page by page, splits them into overlapping chunks, embeds them, and stores them in a persistent ChromaDB collection with `source` (file name) and `page` (1-based) metadata.
2. **Retrieve** – embeds a question and pulls the top-k most similar chunks from ChromaDB, with distance scores.
3. **Augment** – builds a prompt from the retrieved chunks, each labelled with its file and page, and instructs the model to answer only from that context and to cite `(file, page)`.
4. **Generate** – sends the prompt to a local Llama 3.2 3B model through Ollama. If the context does not contain the answer, the model is told to reply exactly `I don't know based on the provided documents.`
5. **Evaluate** – a small script measures retrieval hit rate, out-of-corpus refusal rate, and compares RAG answers with no-context answers.

```
PDFs ──► pypdf ──► RecursiveCharacterTextSplitter ──► HuggingFace embeddings ──► ChromaDB
                                                                                   │
question ──► embed ──► top-k similarity search ◄───────────────────────────────────┘
                              │
                              ▼
                 prompt (context + [file, page] labels)
                              │
                              ▼
                 Llama 3.2 3B (Ollama) ──► answer with citations
```

## Corpus

Six documents (78 pages, 388 chunks):

| Expected file name | Document |
|---|---|
| `Dense_Net.pdf` | DenseNet – [arXiv 1608.06993](https://arxiv.org/abs/1608.06993) |
| `Efficient_Net.pdf` | EfficientNet – [arXiv 1905.11946](https://arxiv.org/abs/1905.11946) |
| `Grad_CAM.pdf` | Grad-CAM – [arXiv 1610.02391](https://arxiv.org/abs/1610.02391) |
| `LIME.pdf` | LIME, "Why Should I Trust You?" – [arXiv 1602.04938](https://arxiv.org/abs/1602.04938) |
| `integrated_axiomatic.pdf` | Integrated Gradients, "Axiomatic Attribution for Deep Networks" – [arXiv 1703.01365](https://arxiv.org/abs/1703.01365) |
| `NGNDAI-2026_Paper_625.pdf` | The author's own paper on XAI for colorectal histology (not included) |

The PDFs are not committed to this repository (`data/` is git-ignored). Download the papers from arXiv and save them in `data/` under the file names above. The sixth document is not public: use any PDF of your own, and update the last in-corpus question in `eval.py` to match it.

## Configuration

| Setting | Value |
|---|---|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Vector store | ChromaDB, collection `Research_Papers`, persisted in `./chroma_db` |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 characters, 150 overlap |
| Retrieval | top-k similarity search, k = 5 (Chroma returns distances: lower is closer) |
| LLM | `llama3.2:3b` through Ollama, temperature 0 |

## Setup

Requires Python 3.12 and [Ollama](https://ollama.com).

```bash
git clone https://github.com/2602-AyushG/RAGForge.git
cd RAGForge

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2:3b     # make sure the Ollama server is running (localhost:11434)

mkdir data                  # then add the PDFs listed above
```

## Usage

**1. Ingest the documents** (the first run embeds all chunks and takes a few minutes on CPU):

```bash
python ingest.py
```

It ends with a proof report: number of chunks stored in Chroma, chunks per source file, chunk length statistics, a count of near-empty chunks, and three random stored chunks to inspect by eye.

**2. Ask a question:**

```bash
python query.py "How does Grad-CAM produce its heatmap?"
```

The output shows the retrieved chunks (file, page, distance score, preview) followed by the final answer.

**3. Run the evaluation:**

```bash
python eval.py
```

## Evaluation

The test set is 13 hand-written questions: 10 in-corpus questions, each with an expected source paper, and 3 out-of-corpus questions.

| Metric | Result |
|---|---|
| Hit@5 (expected paper appears in the top 5 chunks) | 10 / 10 |
| Refusal rate on out-of-corpus questions | 3 / 3 |
| Questions / k | 13 / 5 |
| LLM | Llama 3.2 3B (Ollama) |

`eval.py` also prints RAG and no-context answers side by side for the first three questions, so you can compare them by eye.

**How to read these numbers:** this is a sanity check, not a benchmark. The test set is small, and most in-corpus questions name the method directly ("What is DenseNet?"), which makes retrieval easy. Hit@5 only checks that the right paper is retrieved, not that the right page or the right answer is. The refusal check is an exact string match on questions unrelated to the corpus.

## Project structure

```
ingest.py           PDFs -> chunks -> embeddings -> ChromaDB, plus a proof report
query.py            retrieve -> augment prompt -> generate (CLI)
eval.py             Hit@k, out-of-corpus refusal rate, RAG vs no-context comparison
semantic_search.py  early learning script: in-memory semantic search over a sample PDF
chroma_basics.py    early learning script: persistent ChromaDB basics
requirements.txt    pinned dependencies
```

## Design notes and known limitations

- **Sequential chunk IDs.** Chunks are stored with IDs `chunk_0 … chunk_N`, so re-running `ingest.py` on the same files overwrites instead of duplicating. If you add or remove PDFs, or change the chunk size or overlap, delete `chroma_db/` and re-ingest.
- **PDF extraction noise.** Text is extracted with `pypdf`. On these two-column papers this produces some noise: ligatures kept as single characters (`ﬁ`, `ﬂ`), words hyphenated across line breaks, footnotes interleaved with body text, flattened equations and tables, and reference lists indexed as ordinary chunks.
- **Character-based chunking.** Chunks can start mid-sentence, and each chunk stays within a single page.
- **Small local model.** Llama 3.2 3B runs offline at no cost, but it is weaker than large hosted models, especially on multi-paper or math-heavy questions.
- **Single-turn CLI.** No conversation memory, reranking, or hybrid keyword search.

Ideas for next steps: exclude reference sections from indexing, use section-aware chunking, grow the eval set with paraphrased and cross-paper questions and page-level checks, score answer correctness (not just retrieval), and skip already-stored chunks on re-ingestion.
