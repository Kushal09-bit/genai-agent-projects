FROM python:3.13-slim

WORKDIR /app

# poppler-utils: pdf2image needs pdftoppm/pdfinfo. tesseract-ocr: pytesseract
# needs the tesseract binary. Both are required by src/ingest.py's OCR pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranking models at build time so the image
# works without network access at runtime (ingest/agent/mcp_server/evaluate
# all load the embedding model; mcp_server's retrieve_chunks reranker loads
# the cross-encoder).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src ./src
COPY main.py streamlit_app.py ./

VOLUME ["/app/chroma_db", "/app/data"]

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
