# pdf_loader.py
import pypdf

def load_pdf_text_from_path(pdf_path: str) -> str:
    """Extract text from a PDF file on disk."""
    reader = pypdf.PdfReader(pdf_path)
    text_chunks = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    return "\n".join(text_chunks)

def load_pdf_text_from_memory(pdf_bytes: bytes) -> str:
    """Extract text from an in-memory PDF (uploaded file)."""
    reader = pypdf.PdfReader(pdf_bytes)
    text_chunks = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    return "\n".join(text_chunks)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks of a given size."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
