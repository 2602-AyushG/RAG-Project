"""
Part 2: store the same chunks in Chroma (saved on disk) and query them back.

Run this file TWICE:
  1st run -> embeds all chunks and saves them to ./chroma_db (slow)
  2nd run -> finds them already saved and skips straight to querying (fast)
"""

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/nke-10k-2023.pdf"
DB_DIR = "chroma_db"  # already listed in .gitignore
COLLECTION = "nike_10k"

# Same embedding model as Part 1. Chroma needs it to embed chunks when
# you add them AND to embed your question when you search.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# ---------------------------------------------------------------
# Open (or create) a collection saved in the chroma_db/ folder.
# A collection is like a table: each record holds an id, the chunk
# text, its embedding vector, and its metadata.
# ---------------------------------------------------------------
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

# _collection is the underlying Chroma collection; .count() = number of records.
count = vector_store._collection.count()
print(f"[chroma] collection '{COLLECTION}' currently holds {count} chunks")


# ---------------------------------------------------------------
# INGEST (only if the collection is empty).
# Without this check, every run would add the same 516 chunks again,
# and searches would return the same chunk several times over.
# ---------------------------------------------------------------
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


if count == 0:
    docs = load_pdf_pages(PDF_PATH)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = splitter.split_documents(docs)
    print(f"[ingest] embedding and storing {len(all_splits)} chunks (slow, one time)...")
    vector_store.add_documents(documents=all_splits)
    print(f"[ingest] done, collection now holds {vector_store._collection.count()} chunks")
else:
    print("[ingest] chunks already stored on disk, skipping (no re-embedding)")


# ---------------------------------------------------------------
# PEEK at one stored record, to see exactly what Chroma keeps.
# ---------------------------------------------------------------
record = vector_store.get(limit=1, include=["documents", "metadatas", "embeddings"])
print("\n--- one stored record ---")
print("id:            ", record["ids"][0])
print("metadata:      ", record["metadatas"][0])
print("text:          ", record["documents"][0][:150].replace("\n", " "), "...")
print("embedding size:", len(record["embeddings"][0]))


# ---------------------------------------------------------------
# QUERY. Chroma scores are DISTANCES: lower = closer = better.
# (Part 1's in-memory store used similarity, where higher was better.)
# ---------------------------------------------------------------
queries = [
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
    "What is the capital of France?",  # off-topic on purpose
]

for query in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    results = vector_store.similarity_search_with_score(query, k=3)
    for rank, (doc, distance) in enumerate(results, start=1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n  #{rank}  distance={distance:.3f}  page={doc.metadata['page']}")
        print(f"  {preview}...")


# ---------------------------------------------------------------
# METADATA FILTER: search only chunks whose metadata matches.
# This is something a plain numpy search doesn't give you for free.
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("FILTERED QUERY (page 26 only): distribution centers")
filtered = vector_store.similarity_search_with_score(
    "distribution centers", k=3, filter={"page": 26}
)
for rank, (doc, distance) in enumerate(filtered, start=1):
    print(f"  #{rank}  distance={distance:.3f}  page={doc.metadata['page']}")