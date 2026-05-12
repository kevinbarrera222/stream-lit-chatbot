from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st

#Configurar la pagina de la app 
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖Chatbot Básico con Lanchaing")
st.markdown("Este es un *chatbot de ejemplo* construido con Langchain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


