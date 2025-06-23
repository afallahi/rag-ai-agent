"""PDF Text Extractor Module"""
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = []
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}") from e
