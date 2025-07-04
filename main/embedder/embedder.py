"""Embedding Generator Module"""
import os
from typing import List
from sentence_transformers import SentenceTransformer

# Load the model once (cached)
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks.

    Args:
        chunks (List[str]): List of text strings.

    Returns:
        List[List[float]]: Corresponding list of embeddings.
    """
    if not chunks:
        return []
    
    show_progress = os.getenv("DEBUG", "false").lower() == "true"
    return _model.encode(chunks, convert_to_numpy=True, show_progress_bar=show_progress).tolist()


def get_model() -> SentenceTransformer:
    """
    Expose the internal model (used for query embedding).

    Returns:
        SentenceTransformer: Preloaded embedding model.
    """
    return _model