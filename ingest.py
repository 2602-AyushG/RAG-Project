import pypdf
import random
from collections import Counter
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_dir = Path("papers")
route = sorted(data_dir.glob("*.pdf"))
docs=[]
paths= []
for r in route:
    paths.append(Path(r))
    
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path.name, "page": i+1},
        )
        for i, page in enumerate(reader.pages)
    ]

 
count = 0
for path in paths:
    pages = load_pdf_pages(path)
    docs.extend(pages)
    print(f"[load] {path.name}: {len(pages)} pages") 
    count += len(pages)

print(f"[load] {count} are the total pages loaded")


textsplitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 150, add_start_index=True
)

splitChunks = textsplitter.split_documents(docs)
print(f"[split] {len(splitChunks)} chunks created")

for chunk in splitChunks:
    print("Source:", chunk.metadata["source"])
    print("Page:", chunk.metadata["page"])
    print("Text:", chunk.page_content[:100])
    print("--------------------")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

vector_store = Chroma(
    collection_name = "Research_Papers",
    embedding_function= embeddings,
    persist_directory="./chroma_db"
)

ids = []
for i in range(len(splitChunks)):
    ids.append(f"chunk_{i}")

print("Starting embedding...")

vector_store.add_documents(
    documents = splitChunks,
    ids = ids
)

print("Embedding completed!")

print("Chroma count:", vector_store._collection.count())
data = vector_store.get()

from collections import Counter
import random


# =========================
# 3. Chunks per source file
# =========================

data = vector_store.get()

sources = [metadata["source"] for metadata in data["metadatas"]]

counts = Counter(sources)

print("\n--- Chunks per source file ---")

for source, count in counts.items():
    print(source, ":", count, "chunks")


# =========================
# 4. Shortest, average and longest chunk
# =========================

documents = data["documents"]

lengths = [len(doc) for doc in documents]

shortest = min(lengths)
average = sum(lengths) / len(lengths)
longest = max(lengths)

print("\n--- Chunk lengths ---")

print("Shortest:", shortest)
print("Average:", average)
print("Longest:", longest)


# =========================
# 5. Nearly empty chunks
# =========================

# Here we consider chunks with fewer than 50 characters
# as nearly empty.

nearly_empty = [doc for doc in documents if len(doc.strip()) < 50]

print("\n--- Nearly empty chunks ---")

print("Nearly empty:", len(nearly_empty))


# =========================
# 6. Three random stored chunks
# =========================

print("\n--- Three random chunks ---")

random_chunks = random.sample(
    range(len(documents)),
    min(3, len(documents))
)

for i in random_chunks:

    print("\nSource:", data["metadatas"][i]["source"])
    print("Page:", data["metadatas"][i]["page"])
    print("Text:", documents[i])
    print("--------------------")
