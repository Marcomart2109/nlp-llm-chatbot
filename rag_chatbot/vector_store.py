from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from config import Config as cfg

class VectorStoreManager:
    def __init__(self, documents_path=cfg.DOCUMENTS_PATH, vector_store_path=cfg.VECTOR_STORE_PATH):
        self.documents_path = documents_path
        self.vector_store_path = vector_store_path
        self.docs = []
        self.chunks = []
        self.vector_store = None

    def _load_documents(self):
        pdf_files = list(Path(self.documents_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                self.docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        print(f"Successfully loaded {len(self.docs)} documents.")


    def _split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
        self.chunks = text_splitter.split_documents(self.docs)
        print(f'Before split: {len(self.docs)} pages, after split: {len(self.chunks)} chunks.')
        # Aggiungi questo per controllare i chunk
        for i, chunk in enumerate(self.chunks[:5]):  # Controlla solo i primi 5 per esempio
            print(f"Chunk {i}: {chunk.page_content[:300]}")

    def load_or_generate_vector_store(self):
        if Path(self.vector_store_path).exists():
            print(f'Loading existing vector store from {self.vector_store_path}')
            self.vector_store = FAISS.load_local(self.vector_store_path, HuggingFaceBgeEmbeddings(
                model_name=cfg.HUGGINGFACE_MODEL_NAME,
                encode_kwargs={'normalize_embeddings': True},
            ),
            allow_dangerous_deserialization=True
            )
        else:
            print('Vector store not found. Generating new one...')
            self._load_documents()
            self._split_documents()
            self._generate_vector_store()

    def _generate_vector_store(self):
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=cfg.HUGGINGFACE_MODEL_NAME,
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = FAISS.from_documents(self.chunks, embedding_model)
        self.vector_store.save_local(self.vector_store_path)
        print(f'Vector store saved to {self.vector_store_path}')

# Example usage:
if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.load_or_generate_vector_store()

    # Test search
    if manager.vector_store:
        #print(manager.vector_store.search('Who is the professor of the course?', top_k=1, search_type='similarity'))
        while True:
            query = input("Enter a query (q to quit): ")
            if query == 'q':
                break
            docs = manager.vector_store.similarity_search(query, top_k=5)
            for doc in docs:
                print(f'Source: {doc.metadata['source']}, Page: {doc.metadata["page"]}: {doc.page_content[:1000]}\n')