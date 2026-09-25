"""
Part 1: semantic search over a PDF with LangChain.

Pipeline: LOAD -> SPLIT -> EMBED -> STORE -> QUERY
(No chat model / LLM is needed yet. That comes in Part 4.)
"""

import pypdf
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/nke-10k-2023.pdf"


# 1. LOAD: read the PDF, one Document per page.
#    A Document = page_content (text) + metadata (source, page).
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


docs = load_pdf_pages(PDF_PATH)
print(f"[load]  {len(docs)} pages loaded")


# 2. SPLIT: cut pages into ~1000-character chunks. chunk_overlap=200
#    repeats the end of one chunk at the start of the next so sentences
#    on a boundary keep their context. (You'll break these in Part 5.)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"[split] {len(all_splits)} chunks created")


# 3. EMBED: turn each chunk into a vector (numbers capturing meaning).
#    This model runs locally and downloads on first use.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)
sample_vector = embeddings.embed_query(all_splits[0].page_content)
print(f"[embed] each chunk becomes a vector of length {len(sample_vector)}")
print(f"        first 5 numbers: {sample_vector[:5]}")


# 4. STORE: chunks + vectors go into a vector store.
#    In-memory for now; Part 2 swaps this for Chroma.
vector_store = InMemoryVectorStore(embeddings) 
ids = vector_store.add_documents(documents=all_splits)
print(f"[store] {len(ids)} chunks stored")


# 5. QUERY: embed the question, return the closest chunks.
queries = [
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
    "What was Nike's revenue in 2023?",
    "What is the capital of France?",  # off-topic on purpose
]

for query in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    results = vector_store.similarity_search_with_score(query, k=3)
    for rank, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"\n  #{rank}  score={score:.3f}  page={doc.metadata['page']}")
        print(f"  {preview}...")