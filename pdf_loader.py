# pdf_loader.py
import pypdf

def load_pdf_text(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    pdf_reader = pypdf.PdfReader(pdf_path)
    text_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    return "\n".join(text_chunks)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
