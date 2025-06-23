"""Test suite for the text chunking functionality."""
from main.chunker.text_chunker import chunk_text

def test_chunk_text_basic():
    """Test text is chunked properly."""
    long_text = (
        "This is a test document. " * 100 +  # ~2600 characters
        "Second part of the document. " * 50
    )

    chunks = chunk_text(long_text, chunk_size=500, chunk_overlap=50)

    # Basic validations
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 1, "Expected multiple chunks for long input."

    # Check if chunk size respects the max limit
    assert all(len(chunk) <= 500 for chunk in chunks), "Chunk exceeds max size."
