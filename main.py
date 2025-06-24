"""RAG Project Main Module"""

import os
from main.extractor import pdf_extractor
from main.chunker import text_chunker
from main.embedder import embedder


SAMPLE_DIR = "sample_pdfs"
DEBUG_OUTPUT_DIR = "debug_chunks"

os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)


def process_pdf(file_path: str):
    filename = os.path.basename(file_path)
    print(f"\n--- Processing: {filename} ---")

    text = pdf_extractor.extract_text_from_pdf(file_path)
    if not text.strip():
        print(f"No text extracted from {filename}")
        return

    print(f"Extracted {len(text)} characters")

    chunks = text_chunker.chunk_text(text)
    if not chunks:
        print(f"No chunks created for {filename}")
        return

    avg_size = sum(len(c) for c in chunks) // len(chunks)
    print(f"Created {len(chunks)} chunks (avg size: {avg_size} chars)")
    print("\n First chunk preview:\n")
    print(chunks[0][:500])

    # Save chunks to debug markdown
    debug_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.md")
    with open(debug_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n--- Chunk {i} ---\n{chunk}\n")

    print(f"Chunks saved to: {debug_path}")

    # Step 3: Generate embeddings
    embeddings = embedder.embed_text_chunks(chunks)
    if not embeddings:
        print("No embeddings created.")
        return

    print(f"Generated {len(embeddings)} embeddings (vector size: {len(embeddings[0])})")
    print("First embedding preview (first 10 dims):")
    print(embeddings[0][:10])

    # Save embeddings to a debug file
    debug_embed_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.embeddings.txt")
    with open(debug_embed_path, "w", encoding="utf-8") as f:
        for i, emb in enumerate(embeddings, start=1):
            f.write(f"Embedding {i}: {emb}\n")

    print(f"Embeddings saved to: {debug_embed_path}")


def main():
    """Main"""
    pdf_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found.")
        return
    
    for file in pdf_files:
        file_path = os.path.join(SAMPLE_DIR, file)
        process_pdf(file_path)
    

if __name__ == "__main__":
    main()
