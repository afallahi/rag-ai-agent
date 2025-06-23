"""RAG Project Main Module"""

from main.extractor import pdf_extractor

def main():
    """Main"""
    file_path = "sample_pdfs/dummy.pdf"
    text = pdf_extractor.extract_text_from_pdf(file_path)
    print(f"Extracted {len(text)} characters:\n")
    print(text[:1000])

if __name__ == "__main__":
    main()
