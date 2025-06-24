"""Test suite for embedding generation from text chunks."""

from main.embedder import embedder

def test_embed_text_chunks():
    """Test the embedding generation."""
    sample_chunks = [
        "This is the first test chunk.",
        "Another chunk of text to embed.",
        "Yet another meaningful text segment."
    ]

    embeddings = embedder.embed_text_chunks(sample_chunks)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(sample_chunks)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(dim, float) for vec in embeddings for dim in vec)

    # All vectors should be the same length
    dim_lengths = set(len(vec) for vec in embeddings)
    assert len(dim_lengths) == 1, "Inconsistent embedding dimensions"
