import os
from dotenv import load_dotenv

# Load environment variables in .env file
load_dotenv()

class Config:
    # Get environment variables
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    MIN_CHUNK_LENGTH = 50
    TOP_K = 5
    DOCUMENTS_PATH = "data/slides_and_syllabus"
    VECTOR_STORE_PATH = "data/vector_store"
    LLM_MODEL_NAME="mistral-large-latest"
    LLM_MODEL_PROVIDER="mistralai"
    MISTRAL_EMBEDDING_MODEL="mistral-embed"
    HF_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
    TEMPERATURE=0.3
