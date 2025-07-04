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
├── sample_pdfs/
├── main.py
├── main/
│ └── extractor.py
│   └── pdf_extractor.py # Step 1: PDF extraction
│ └── chunker/
│   └── text_chunker.py # Step 2: Text chunking
│ └── embedder/
│   └── embedder.py # Step 3: Embedding
│ └── vector_store/
│   └── faiss_indexer.py # Step 4: FAISS vector DB
│ └── llm/
│   └── llm_client.py # Step 5: LLM Integration (Ollama)
├── tests/
│ └── test_pdf_extractor.py
│ └── test_text_chunker.py
│ └── test_embedder.py
│ └── test_vector_store.py
│ └── test_llm_client.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Conda environment

```bash
conda create -n rag python=3.11
```


### 2. Activate environment

`conda activate rag`

### 3. Install requirements

`pip install -r requirements.txt`


### 4. Install and Run Ollama (for local LLM)
Ollama is used to run local LLMs like Mistral.

Download and install [Ollama](https://ollama.com)
Pull and run a model (e.g., `mistral`)

```bash

ollama pull mistral
ollama run mistral
```

## Running the Project

`python main.py`

### Force Reprocessing
By default, the system skips PDFs that already have a processed FAISS index. To force reprocessing of all PDFs:
`python main.py --force`


## Steps

| Step | Description                         | Status          |
| ---- | ----------------------------------- | --------------  | 
| 1    | PDF Text Extraction (PyMuPDF)       | ✅ Completed    |
| 2    | Text Chunking (LangChain)           | ✅ Completed    |
| 3    | Embedding with SentenceTransformers | ✅ Completed    |
| 4    | Vector Store Setup (FAISS)          | ✅ Completed    |
| 5    | LLM Integration (Ollama)            | ✅ Completed    |
| 6    | Full RAG Pipeline                   | ✅ Completed    |


## Running Tests

`pytest`

## Tools Used

- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for PDF parsing
- LangChain for chunking logic
- [Sentence Transformers](https://www.sbert.net/) for embedding
- [FAISS](https://github.com/facebookresearch/faiss) Facebook AI Similarity Search for vector search
- [Ollama](https://ollama.com/) for running local LLMs like Mistral.
