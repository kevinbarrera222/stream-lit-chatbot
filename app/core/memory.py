# app/core/memory.py

from langchain.schema import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class LongTermMemory:
    def __init__(self, persist_directory="memory_db"):
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(
            collection_name="chat_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def save_memory(self, text: str):
        """Guarda un recuerdo en la base vectorial."""
        self.db.add_texts([text])
        self.db.persist()

    def retrieve_memory(self, query: str, k: int = 3):
        """Recupera los recuerdos más relevantes."""
        results = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

def historial_a_texto(mensajes, limite=10):
    """Convierte el historial reciente en texto plano."""
    historial = ""
    for msg in mensajes[-limite:]:
        if isinstance(msg, HumanMessage):
            historial += f"Usuario: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            historial += f"Asistente: {msg.content}\n"
    return historial or "(No hay historial previo)"
