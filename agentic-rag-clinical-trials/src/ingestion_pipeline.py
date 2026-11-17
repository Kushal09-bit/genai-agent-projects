

import glob
import hashlib
import os
import re

import chromadb
import pymupdf
from chromadb.utils import embedding_functions

from src.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_PATH,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBED_MODEL_NAME,
    MAX_HEADING_WORDS,
)

# OCR fallback only - most academic PDFs have a real text layer and never touch this
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# numbered headings like "2.1 Methods" - capped length + no comma so it doesn't
# eat numbered reference-list entries ("9. Okokhere P, Colubri A, et al...")
HEADING_REGEX = re.compile(r"^\s*(\d+(\.\d+)*)\.?\s+[A-Z][A-Za-z][^,]{0,55}$")


# ---------- text + heading extraction ----------

def _get_body_font_size(doc):
    """most common font size in the doc = body text. headings deviate from this."""
    sizes = {}
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    sizes[size] = sizes.get(size, 0) + len(span["text"])
    return max(sizes, key=sizes.get) if sizes else 10.0


def _is_heading_line(text, spans, body_size):
    text = text.strip()
    if not text or len(text) > 90:
        return False
    if HEADING_REGEX.match(text):
        return True
    if len(text) < 3:
        return False  # drop-cap first letters showing up as their own oversized span

    max_size = max(round(s["size"], 1) for s in spans)
    if max_size > body_size + 1.0 and len(text) < 100:
        return True

    # named headings (Methods, Results...) are usually bold but NOT bigger than body text -
    # catch those by shape: short, standalone, bold, title-cased line
    all_bold = all("bold" in s["font"].lower() for s in spans)
    word_count = len(text.split())
    if (
        all_bold
        and max_size >= body_size - 1.5
        and 1 <= word_count <= MAX_HEADING_WORDS
        and text[0].isupper()
        and "," not in text
        and not text.endswith((",", ";"))
    ):
        return True
    return False


def _ocr_page(filepath, page_number):
    """fallback for a page with no extractable text - only runs if PyMuPDF got nothing."""
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_path(filepath, first_page=page_number, last_page=page_number)
        return pytesseract.image_to_string(images[0]) if images else ""
    except Exception as e:
        print(f"  OCR fallback failed for {filepath} page {page_number}: {e}")
        return ""


def extract_sections(filepath):
    """
    walks the pdf and returns a list of sections, each with a heading_path
    (the stack of headings above it), its text, and which pages it spans.
    a chunk built from one of these can never cross a section boundary,
    since sections ARE the boundary.
    """
    doc = pymupdf.open(filepath)
    body_size = _get_body_font_size(doc)

    lines = []
    for page_num, page in enumerate(doc, start=1):
        page_had_text = False
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                page_had_text = True
                lines.append({
                    "text": text,
                    "page": page_num,
                    "is_heading": _is_heading_line(text, spans, body_size),
                })
        if not page_had_text:
            ocr_text = _ocr_page(filepath, page_num)
            for line_text in ocr_text.splitlines():
                if line_text.strip():
                    lines.append({"text": line_text.strip(), "page": page_num, "is_heading": False})
    doc.close()
    return _group_into_sections(lines)


def _group_into_sections(lines):
    sections = []
    heading_stack = []
    current_text, current_pages = [], set()

    def flush():
        if current_text:
            sections.append({
                "heading_path": list(heading_stack) if heading_stack else ["(preamble)"],
                "text": "\n".join(current_text).strip(),
                "pages": sorted(current_pages),
            })

    for line in lines:
        if line["is_heading"]:
            flush()
            current_text, current_pages = [], set()
            m = HEADING_REGEX.match(line["text"])
            depth = m.group(1).count(".") + 1 if m else 1
            heading_stack = heading_stack[: depth - 1]
            heading_stack.append(line["text"])
        else:
            current_text.append(line["text"])
            current_pages.add(line["page"])

    flush()
    return [s for s in sections if len(s["text"]) > 20]  # drop empty headings, stray page numbers etc


# ---------- chunking ----------

def _tokenize(text):
    return text.split()


def _chunk_id(source_doc, heading_path, chunk_index, text):
    raw = f"{source_doc}|{'>'.join(heading_path)}|{chunk_index}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def chunk_section_text(text, size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    tokens = _tokenize(text)
    if len(tokens) <= size:
        return [" ".join(tokens)]
    pieces, start = [], 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        pieces.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return pieces


def load_pdfs(data_dir):
    """walks docs/, chunks every pdf section-aware, returns the flat chunk list."""
    all_chunks = []
    for filepath in glob.glob(os.path.join(data_dir, "*.pdf")):
        filename = os.path.basename(filepath)
        print(f"processing {filename}...")
        sections = extract_sections(filepath)

        idx = 0
        for section in sections:
            for piece in chunk_section_text(section["text"]):
                all_chunks.append({
                    "id": _chunk_id(filename, section["heading_path"], idx, piece),
                    "text": piece,
                    "document_id": filename.replace(".pdf", ""),
                    "heading_path": " > ".join(section["heading_path"]),
                    "pages": section["pages"],
                    "chunk_index": idx,
                    "token_count": len(_tokenize(piece)),
                })
                idx += 1
        print(f"  -> {idx} chunks")
    return all_chunks


# ---------- embed + store ----------

def build_chroma_index(chunks):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    # wipe and recreate so re-running this doesn't leave stale/duplicate chunks around
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedder)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "document_id": c["document_id"],
            "heading_path": c["heading_path"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
        } for c in chunks],
    )
    return collection


if __name__ == "__main__":
    print("Loading + section-aware chunking PDFs...")
    chunks = load_pdfs(str(DOCS_DIR))
    print(f"Created {len(chunks)} chunks total.")
    print("Embedding + storing in ChromaDB...")
    build_chroma_index(chunks)
    print("Done.")
