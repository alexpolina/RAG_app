# pdf_loader.py
import pypdf
from io import BytesIO  

def load_pdf_text_from_memory(pdf_bytes: bytes) -> str:
    """Extracts text from an in-memory PDF using pypdf."""
    pdf_stream = BytesIO(pdf_bytes)  
    reader = pypdf.PdfReader(pdf_stream)

    text_chunks = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    return "\n".join(text_chunks)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Splits text into overlapping chunks of a given size."""
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
