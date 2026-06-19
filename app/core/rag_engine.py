# app/core/rag_engine.py

import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

class RAGEngine:
    def __init__(self, persist_directory="rag_db"):
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_pdf(self, pdf_path: str):
        """Carga un PDF, lo divide en páginas y lo guarda en la base vectorial."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"El archivo no existe: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        texts = [page.page_content for page in pages]
        self.db.add_texts(texts)
        self.db.persist()

        return len(texts)

    def search(self, query: str, k: int = 4):
        """Busca información relevante en los documentos."""
        results = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
