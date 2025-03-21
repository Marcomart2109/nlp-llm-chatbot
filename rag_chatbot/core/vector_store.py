import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from pathlib import Path
from config import Config as cfg

class VectorStoreManager:
    '''
    Singleton class that manages the vector store.'''
    _instance = None # instance of the class

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialized = False  # Flag to check if the class has been initialized
        return cls._instance

    def __init__(self, documents_path=cfg.DOCUMENTS_PATH, vector_store_path=cfg.VECTOR_STORE_PATH):
        if not self._initialized:
            self.documents_path = documents_path
            self.vector_store_path = vector_store_path
            self.docs = []
            self.chunks = []
            self.vector_store = None
            # Mistral AI embeddings are norm 1 (cosine similarity, dot product or Euclidean distance are all equivalent).
            # self.embeddings = MistralAIEmbeddings(model=cfg.MISTRAL_EMBEDDING_MODEL)
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=cfg.HF_EMBEDDING_MODEL,
                encode_kwargs={'normalize_embeddings': True}
            )
            self._initialized = True  # Set the flag to True

    def _load_documents(self):
        pdf_files = list(Path(self.documents_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            filename = pdf_file.name
            # We will esclude the first and the last page only for the slides
            try:
                # Check if the filename starts with a number
                starts_with_number = filename[0].isdigit()
                
                if starts_with_number:
                    # For documents starting with a number, load and skip first and last page
                    import pdfplumber  # Import here to avoid circular imports
                    
                    # First, count the total pages
                    with pdfplumber.open(str(pdf_file)) as pdf:
                        total_pages = len(pdf.pages)
                        if total_pages <= 2:  # Not enough pages to skip first and last
                            print(f"Warning: {filename} has only {total_pages} pages, need at least 3 to skip first and last. Loading all pages.")
                            loader = PDFPlumberLoader(str(pdf_file))
                            self.docs.extend(loader.load())
                        else:
                            # Create a custom loader for specific pages (skip first and last)
                            pages = []
                            with pdfplumber.open(str(pdf_file)) as pdf:
                                for i in range(1, total_pages - 1):  # Skip page 0 and the last page
                                    page = pdf.pages[i]
                                    text = page.extract_text()
                                    # Create a Document object similar to what PDFPlumberLoader would create
                                    from langchain_core.documents import Document
                                    doc = Document(
                                        page_content=text,
                                        metadata={
                                            "source": str(pdf_file),
                                            "page": i,
                                            "total_pages": total_pages
                                        }
                                    )
                                    pages.append(doc)
                            self.docs.extend(pages)
                            print(f"Loaded {filename} - skipped first and last pages ({len(pages)} pages loaded)")
                else:
                    # For other documents, load all pages
                    loader = PDFPlumberLoader(str(pdf_file))
                    loaded_docs = loader.load()
                    self.docs.extend(loaded_docs)
                    print(f"Loaded {filename} - all pages ({len(loaded_docs)} pages loaded)")
                    
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        
        print(f"Successfully loaded {len(self.docs)} documents.")

    def _split_documents(self):
        """Split documents into chunks and filter out chunks that are too short."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
        all_chunks = text_splitter.split_documents(self.docs)
        
        # Filter chunks by minimum length
        self.chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= cfg.MIN_CHUNK_LENGTH]
        
        print(f'Before split: {len(self.docs)} pages, after split: {len(all_chunks)} chunks.')
        print(f'After length filtering: {len(self.chunks)} chunks (minimum {cfg.MIN_CHUNK_LENGTH} characters).')

    def load_or_generate_vector_store(self):
        if Path(self.vector_store_path).exists():
            print(f'Loading existing vector store from {self.vector_store_path}')
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            print(f'Vector store successfully loaded from {self.vector_store_path}')
        else:
            print('Vector store not found. Generating new one...')
            self._load_documents()
            self._split_documents()
            self._generate_vector_store()

    def _generate_vector_store(self):
        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f'Vector store saved to {self.vector_store_path}')

# # Example usage:
# if __name__ == "__main__":
#     manager = VectorStoreManager()
#     manager.load_or_generate_vector_store()

#     # Test search
#     if manager.vector_store:
#         #print(manager.vector_store.search('Who is the professor of the course?', top_k=1, search_type='similarity'))
#         while True:
#             query = input("Enter a query (q to quit): ")
#             if query == 'q':
#                 break
#             docs = manager.vector_store.similarity_search(query, top_k=3)
#             for doc in docs:
#                 print(f'\nSource: {doc.metadata['source']}, Page: {doc.metadata["page"]}: {doc.page_content}\n\n')