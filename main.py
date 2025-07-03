"""RAG Project Main Module"""

import os
import logging
from main.extractor import pdf_extractor
from main.chunker import text_chunker
from main.embedder import embedder
from main.vector_store import faiss_indexer
from main.llm import llm_client



# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SAMPLE_DIR = "sample_pdfs"
DEBUG_OUTPUT_DIR = "debug_chunks"
FAISS_INDEX_DIR = "faiss_index"


os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


def save_debug_outputs(filename: str, chunks: list[str], embeddings: list[list[float]]):
    """Save chunks and embeddings to debug files."""
    # Save chunks
    debug_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.md")
    with open(debug_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n--- Chunk {i} ---\n{chunk}\n")
    logger.info("Chunks saved to: %s", debug_path)

    # Save embeddings
    debug_embed_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.embeddings.txt")
    with open(debug_embed_path, "w", encoding="utf-8") as f:
        for i, emb in enumerate(embeddings, start=1):
            f.write(f"Embedding {i}: {emb}\n")
    logger.info("Embeddings saved to: %s", debug_embed_path)


def process_pdf(file_path: str, query_text: str):
    """Process a single PDF and query its contents."""

    filename = os.path.basename(file_path)
    logger.info("Processing: %s", filename)

    text = pdf_extractor.extract_text_from_pdf(file_path)
    if not text.strip():
        logger.warning("No text extracted from %s", filename)
        return

    logger.debug("Extracted %d characters", len(text))

    chunks = text_chunker.chunk_text(text)
    if not chunks:
        logger.warning("No chunks created for %s", filename)
        return

    logger.info("Created %d chunks", len(chunks))
    logger.debug("First chunk preview:\n%s", chunks[0][:500])

    # Step 3: Generate embeddings
    embeddings = embedder.embed_text_chunks(chunks)
    if not embeddings:
        logger.warning("No embeddings created.")
        return

    logger.info("Generated %d embeddings", len(embeddings))
    logger.debug("First embedding preview:\n%s", embeddings[0][:10])


    # Save embeddings to a debug file
    save_debug_outputs(filename, chunks, embeddings)

    # Step 4: Store embeddings in FAISS
    try:
        logger.info("Building FAISS index...")
        index = faiss_indexer.build_faiss_index(embeddings, chunks)

        # Save FAISS index
        index_path = os.path.join(FAISS_INDEX_DIR, f"{filename}.index")
        faiss_indexer.save_faiss_index(index, index_path)
        logger.debug("FAISS index saved to: %s", index_path)
        
        # Query vector store
        top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedder.get_model(), k=2)
        if not top_chunks:
            logger.info("No matching chunks found for query: %s", query_text)
            return
        
        # Extract scores to check relevance
        max_score = max(score for _, score in top_chunks)
        threshold = 0.2  # Cosine similarity threshold

        if max_score < threshold:
            logger.info("No relevant chunks found for query: '%s'", query_text)
            return
        
        for i, chunk in enumerate(top_chunks, start=1):
            print(f"\nTop Match {i}:\n{chunk[:300]}...")
        logger.info("Retrieved %d top matching chunks for query: '%s'", len(top_chunks), query_text)

        # Step 5: LLM integration
        context = "\n\n".join(chunk for chunk, _ in top_chunks)
        full_prompt = (
            "You are a helpful assistant.\n\n"
            "Answer the question below using ONLY the context provided.\n\n"
            f"Context:\n{context}\n\nQuestion: {query_text}"
        )

        response = llm_client.generate_answer(full_prompt)
        print("\n LLM Response:\n", response)

    except Exception as e:
        logger.error("FAISS index operation failed: %s", e)


def main():
    """Main"""
    pdf_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found.")
        return
    
    query = input("Enter your search query: ")
    if not query:
        logger.warning("Empty query provided. Skipping search.")
        return
    
    for file in pdf_files:
        file_path = os.path.join(SAMPLE_DIR, file)
        process_pdf(file_path, query)
    

if __name__ == "__main__":
    main()
