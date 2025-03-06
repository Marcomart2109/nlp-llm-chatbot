import os
from dotenv import load_dotenv

# Load environment variables in .env file
load_dotenv()

class Config:
    # Get environment variables
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    DOCUMENTS_PATH = "data/documents"
    VECTOR_STORE_PATH = "data/vector_store"
    LLM_MODEL_NAME="mistral-large-latest"
    LLM_MODEL_PROVIDER="mistralai"
    EMBEDDING_MODEL="mistral-embed"
