import os
import sys
import json
import urllib.request

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Same values used in ingest.py
COLLECTION_NAME = "Research_Papers"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

MODEL_NAME = "llama3.2:3b"

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


# Call OpenAI API
# def call_llm(prompt):
#     api_key = os.getenv("OPENAI_API_KEY")

#     data = json.dumps({
#         "model": MODEL_NAME,
#         "temperature": 0,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     }).encode("utf-8")

#     request = urllib.request.Request(
#         "https://api.openai.com/v1/chat/completions",
#         data=data,
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {api_key}"
#         }
#     )

#     with urllib.request.urlopen(request) as response:
#         result = json.loads(
#             response.read().decode("utf-8")
#         )

#     return result["choices"][0]["message"]["content"]

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
        "http://localhost:11434/api/generate",
        data=data,
        headers={
            "Content-Type": "application/json"
        }
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(
            response.read().decode("utf-8")
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


# Run from command line
if __name__ == "__main__":

    query = " ".join(sys.argv[1:])

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

