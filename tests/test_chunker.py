"""Test suite for the text chunking functionality."""
from main.chunker.text_chunker import chunk_text


def test_chunk_text_basic():
    """Test text is chunked properly and includes expected overlaps."""
    long_text = (
        "This is paragraph one. " * 10 +
        "This is paragraph two. " * 10 +
        "This is paragraph three. " * 10
    )

    chunk_size = 500
    chunk_overlap = 50

    chunks = chunk_text(long_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) > 1, "Expected multiple chunks for long input."
    assert all(len(chunk) <= chunk_size for chunk in chunks), "Chunk exceeds max size."

    if len(chunks) > 1:
        first_chunk_tail = chunks[0][-chunk_overlap:]
        second_chunk_start = chunks[1][:chunk_overlap * 2]  # allow a bit of buffer
        similarity = sum(1 for word in first_chunk_tail.split() if word in second_chunk_start)
        assert similarity > 3, f"Not enough overlapping words found. Overlap similarity: {similarity}"

    print(f"Chunk test passed: {len(chunks)} chunks created.")
