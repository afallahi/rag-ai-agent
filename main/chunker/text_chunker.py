"""Text Chunking Module using LangChain."""
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks.

    Args:
        text (str): The full text to split.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Logical fallback order
    )
    return splitter.split_text(text)

    