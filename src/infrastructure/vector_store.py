import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.infrastructure.llm_factory import LLMFactory

class VectorStoreRepository:
    def __init__(self):
        self.embeddings = LLMFactory.get_embeddings()
        self.vectorstore = None
        self._initialize_db()

    def _download_content(self):
        if not os.path.exists(settings.storage_path):
            print(f"📥 Baixando corpus de {settings.book_url}...")
            response = requests.get(settings.book_url)
            response.encoding = 'utf-8'
            with open(settings.storage_path, "w", encoding='utf-8') as f:
                f.write(response.text)
        
        with open(settings.storage_path, "r", encoding='utf-8') as f:
            return f.read()

    def _initialize_db(self):
        text_content = self._download_content()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        docs = splitter.create_documents([text_content])
        
        print("⚙️ Indexando vetores (FAISS)...")
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def get_retriever(self, k: int = 3):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})