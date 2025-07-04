import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    SAMPLE_DIR: str = os.getenv("SAMPLE_DIR", "sample_pdfs")
    DEBUG_OUTPUT_DIR: str = os.getenv("DEBUG_OUTPUT_DIR", "debug_chunks")

    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")