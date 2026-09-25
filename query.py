import argparse
import json
import sys
import urllib.error
import urllib.request

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Same values used in ingest.py
COLLECTION_NAME = "Research_Papers"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# Load existing Chroma database
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return db


# Retrieve top-k chunks
def retrieve(db, query, k=5):
    # Chroma returns a distance score:
    # lower score = more similar
    return db.similarity_search_with_score(query, k=k)


# Build prompt using retrieved chunks
def build_prompt(query, chunks):
    context = ""

    for i, (doc, score) in enumerate(chunks, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        context += f"""
CONTEXT {i}
[{source}, page {page}]
{doc.page_content}
"""

    prompt = f"""
Answer ONLY using the context below.

If the answer is not in the context, reply exactly:
"I don't know based on the provided documents."

Cite claims using (file, page).

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    return prompt


# Call local Ollama model
def call_llm(prompt):
    data = json.dumps({
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}. "
            f"Make sure the Ollama server and model are running."
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at "
            f"{OLLAMA_URL}. Start Ollama and make sure "
            f"'{MODEL_NAME}' is available."
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "Ollama took too long to respond. "
            "Make sure the model is running and try again."
        ) from exc

    if "error" in result:
        raise RuntimeError(
            f"Ollama error: {result['error']}"
        )

    return result["response"]


# Complete RAG pipeline
def generate(db, query):
    chunks = retrieve(db, query)

    prompt = build_prompt(
        query,
        chunks
    )

    answer = call_llm(prompt)

    return answer, chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the research-paper RAG system."
    )

    parser.add_argument(
        "query",
        nargs="+",
        help="Question to ask about the research papers."
    )

    return parser.parse_args()


# Run from command line
if __name__ == "__main__":
    args = parse_args()
    query = " ".join(args.query).strip()

    try:
        db = load_db()

        answer, chunks = generate(
            db,
            query
        )

        print("\n--- RETRIEVED CHUNKS ---")

        for i, (doc, score) in enumerate(chunks, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")

            print(f"\n{i}. {source}, page {page}")
            print(f"Score: {score:.4f}")
            print(doc.page_content[:200])

        print("\n--- FINAL ANSWER ---")
        print(answer)

    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
