"""RAG Project Main Module"""

import os
import logging
import argparse
from main.extractor import pdf_extractor
from main.chunker import text_chunker
from main.embedder import embedder
from main.vector_store import faiss_indexer
from main.config import Config
from main.llm.ollama_client import OllamaClient


# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SAMPLE_DIR = Config.SAMPLE_DIR
DEBUG_OUTPUT_DIR = Config.DEBUG_OUTPUT_DIR
FAISS_INDEX_PATH = os.path.join("faiss_index", "global.index")


os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)


def save_debug_outputs(filename: str, chunks: list[str], embeddings: list[list[float]]):
    """Save chunks and embeddings to debug files."""
    # Save chunks
    debug_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.md")
    with open(debug_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n--- Chunk {i} ---\n{chunk}\n")
    logger.debug("Chunks saved to: %s", debug_path)

    # Save embeddings
    debug_embed_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.embeddings.txt")
    with open(debug_embed_path, "w", encoding="utf-8") as f:
        for i, emb in enumerate(embeddings, start=1):
            f.write(f"Embedding {i}: {emb}\n")
    logger.debug("Embeddings saved to: %s", debug_embed_path)


def build_prompt(context: str, query: str) -> str:
    return (
        "You are a professional HVAC systems consultant. Use ONLY the context below to answer the following customer question.\n"
        "Answer in a concise, informative paragraph. If the context does not contain the answer, say 'The context does not provide enough information.'\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def build_global_index(force: bool = False):
    """Extract, chunk, and embed all PDFs and return a global index."""
    if os.path.exists(FAISS_INDEX_PATH) and not force:
        logger.info("Global index already exists. Skipping reprocessing.")
        return faiss_indexer.load_faiss_index(FAISS_INDEX_PATH)

    all_chunks = []
    all_embeddings = []

    pdf_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found.")
        return None

    for file in pdf_files:
        file_path = os.path.join(SAMPLE_DIR, file)
        logger.debug("Processing: %s", file)

        text = pdf_extractor.extract_text_from_pdf(file_path)
        if not text.strip():
            logger.warning("No text extracted from %s", file)
            continue

        chunks = text_chunker.chunk_text(text)
        if not chunks:
            logger.warning("No chunks created for %s", file)
            continue

        logger.debug("Created %d chunks from %s", len(chunks), file)
        embeddings = embedder.embed_text_chunks(chunks)
        if not embeddings:
            logger.warning("No embeddings created for %s", file)
            continue

        logger.debug("Generated %d embeddings from %s", len(embeddings), file)
        save_debug_outputs(file, chunks, embeddings)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    if not all_chunks or not all_embeddings:
        logger.warning("No data to build global FAISS index.")
        return None

    logger.debug("Building global FAISS index...")
    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, FAISS_INDEX_PATH)
    logger.debug("Global FAISS index saved to: %s", FAISS_INDEX_PATH)

    return index


def query_and_respond(index, query_text: str, llm):
    """Query global index and generate a response using LLM."""
    top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedder.get_model(), k=4)
    if not top_chunks:
        logger.info("No matching chunks found for query: %s", query_text)
        return

    max_score = max(score for _, score in top_chunks)
    threshold = 0.2
    if max_score < threshold:
        logger.info("No relevant chunks found for query: '%s'", query_text)
        return

    for i, (chunk, score) in enumerate(top_chunks, start=1):
        logger.debug("Top Match %d (score=%.3f): %s...", i, score, chunk[:300])

    logger.debug("Retrieved %d top matching chunks for query: '%s'", len(top_chunks), query_text)

    context = "\n\n".join(chunk for chunk, _ in top_chunks)
    prompt = build_prompt(context, query_text)

    logger.info("\nGenerating Answer using the LLM. This may take a few seconds...")
    response = llm.generate_answer(prompt)
    print("\nLLM Response:\n", response)


def main():
    """Main"""

    llm = OllamaClient()
    if not llm.is_running():
        logger.error("Ollama is not running. Please start Ollama before continuing.")
        return
    
    parser = argparse.ArgumentParser(description="Run RAG pipeline on sample PDFs")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if FAISS index exists")
    args = parser.parse_args()

    pdf_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found.")
        return
    
    index = build_global_index(force=args.force)
    if index is None:
        logger.warning("Index could not be created or loaded.")
        return
    
    try:
        while True:
            query = input("Enter your question (or Ctrl+C to exit): ")
            if not query:
                logger.warning("Empty query provided. Skipping search.")
                return
            if not query.strip():
                logger.warning("Empty query. Please enter a question.")
                continue
            
            query_and_respond(index, query, llm)
    except KeyboardInterrupt:
        logger.info("\nExiting on user interrupt")
        

if __name__ == "__main__":
    main()
