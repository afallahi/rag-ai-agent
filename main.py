"""RAG Project Main Module"""

from main.extractor import pdf_extractor
from main.chunker import text_chunker

def main():
    """Main"""
    file_path = "sample_pdfs/dummy.pdf"

    # Step 1: Extract text from PDF
    text = pdf_extractor.extract_text_from_pdf(file_path)
    print(f"[✓] Extracted {len(text)} characters:\n")
    print(text[:1000])

     # Step 2: Chunk the extracted text
    chunks = text_chunker.chunk_text(text, chunk_size=500, chunk_overlap=50)
    print(f"[✓] Created {len(chunks)} chunks.\n")

    if chunks:
        print("Preview of first chunk:\n")
        print(chunks[0][:1000])

if __name__ == "__main__":
    main()
