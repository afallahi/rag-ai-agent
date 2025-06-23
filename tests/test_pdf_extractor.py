"""Test suite for PDF text extraction functionality."""
import os
from main.extractor.pdf_extractor import extract_text_from_pdf

def test_extract_text_from_pdf():
    """Test the PDF text extraction functionality."""
    sample_pdf = "sample_pdfs/dummy.pdf"
    
    # Check the sample file exists
    assert os.path.exists(sample_pdf), f"{sample_pdf} does not exist."

    text = extract_text_from_pdf(sample_pdf)
    
    # Basic tests to ensure PDF is parsed
    assert isinstance(text, str)
    assert len(text) > 100  # Adjust threshold depending on your PDF
    assert "Introduction" in text or "Summary" in text or "the" in text.lower()
