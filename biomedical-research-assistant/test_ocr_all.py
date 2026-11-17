import os
import glob
from pdf2image import convert_from_path
import pytesseract

DATA_DIR = "data/pubmed_docs"

def ocr_pdf(filepath: str, max_pages: int = 1) -> str:
    """Convert first page(s) of a PDF to images and OCR them."""
    images = convert_from_path(filepath, first_page=1, last_page=max_pages)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

for filepath in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"OCR result for: {filename}")
    print('='*60)
    try:
        text = ocr_pdf(filepath, max_pages=1)
        print(text[:400])
    except Exception as e:
        print(f"OCR failed: {e}")