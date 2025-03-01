import os
from dotenv import load_dotenv

# Load environment variables in .env file
load_dotenv()

class Config:
    # Get environment variables
    HUGGINGFACE_MODEL_NAME = os.getenv('HUGGINGFACE_MODEL_NAME')
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
    UNSTRUCTURED_API_KEY = os.getenv('UNSTRUCTURED_API_KEY')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP'))
    DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH')
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH')
