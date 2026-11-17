"""
OCR-ingests the PDFs in data/pubmed_docs/ into the chroma_db/ vector
store that mcp_layer/mcp_server.py reads from at query time.

Only needed when adding/re-processing documents - the multi-agent
app itself just reads the already-built chroma_db/. Deps for this
script live in requirements/rag-ingestion.txt, not the main
requirements files (they're only needed here, not at query time).

Run with: python ingest_biomedical_corpus.py (from the repo root).
"""

import os
import glob
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DATA_DIR = "data/pubmed_docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pubmed_chunks"


def get_page_count(filepath: str) -> int:
    reader = PdfReader(filepath)
    return len(reader.pages)


def ocr_page(filepath: str, page_number: int) -> str:
    try:
        images = convert_from_path(filepath, first_page=page_number, last_page=page_number)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception as e:
        print(f"  OCR failed for {filepath} page {page_number}: {e}")
    return ""


def load_pdfs(data_dir: str) -> list[dict]:
    documents = []
    for filepath in glob.glob(os.path.join(data_dir, "*.pdf")):
        filename = os.path.basename(filepath)
        doc_id = filename.replace(".pdf", "")
        num_pages = get_page_count(filepath)

        print(f"OCR processing {filename} ({num_pages} pages)...")
        for page_num in range(1, num_pages + 1):
            text = ocr_page(filepath, page_num)
            if text.strip():
                documents.append({
                    "document_id": doc_id,
                    "page_number": page_num,
                    "text": text,
                })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{doc['document_id']}_p{doc['page_number']}_c{i}",
                "text": piece,
                "document_id": doc["document_id"],
                "page_number": doc["page_number"],
                "chunk_index": i,
            })
    return chunks


def build_chroma_index(chunks: list[dict]):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"document_id": c["document_id"], "page_number": c["page_number"], "chunk_index": c["chunk_index"]} for c in chunks],
    )
    return collection


if __name__ == "__main__":
    print("Loading PDFs with OCR...")
    docs = load_pdfs(DATA_DIR)
    print(f"Loaded {len(docs)} pages from PDFs.")
    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print("Embedding + storing in ChromaDB...")
    build_chroma_index(chunks)
    print("Done.")