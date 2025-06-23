# RAG PDF Agent

This project builds a Retrieval-Augmented Generation (RAG) system. It is developed in a **test-driven way**, where each step includes both functionality and verification via unit tests.



## Project Goals

- Parse and extract text from PDFs
- Chunk and embed the text
- Store embeddings in a vector database (FAISS)
- Use a local LLM for question-answering (RAG pipeline)
- Validate every stage with automated tests



## Project Structure

```
rag-project/
├── .venv/
├── sample_pdfs/
├── main.py
├── main/
│ └── extractor.py
│   └── pdf_extractor.py # Step 1: PDF extraction
│ └── chunker/
│   └── text_chunker.py # Step 2: Text chunking
├── tests/
│ └── test_pdf_extractor.py
│ └── test_text_chunker.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create virtual environment

Run `python -m venv .venv`

### 2. 2. Activate environment


Run `.\.venv\Scripts\activate`

### 3. Install requirements

`pip install -r requirements.txt`


## Steps

| Step | Description                         | Status          |
| ---- | ----------------------------------- | --------------  | 
| 1    | PDF Text Extraction (PyMuPDF)       | ✅ Completed    |
| 2    | Text Chunking (LangChain)           | ✅ Completed    |
| 3    | Embedding with SentenceTransformers | ⏳ Pending      |
| 4    | Vector Store Setup (FAISS)          | ⏳ Pending      |
| 5    | LLM Integration (Ollama, llama-cpp) | ⏳ Pending      |
| 6    | Full RAG Pipeline                   | ⏳ Pending      |


## Running Tests

`pytest`

## Tools Used

- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for PDF parsing
- LangChain for chunking logic
- Sentence Transformers for embedding
- FAISS for vector search
- Local LLMs (via Ollama or llama-cpp)
