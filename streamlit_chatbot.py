from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st
from langchain.prompts import PromptTemplate 

#Configurar la pagina de la app 
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖Chatbot Básico con Lanchaing")
st.markdown("Este es un *chatbot de ejemplo* construido con Langchain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura",0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo",["gtp-3.5-turbo","gtp-4","gpt-40-mini"])

    # Recrea el modelo con nuevos parametros
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# Inicializar el historial de mensajes
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


 #crear el template de prompt con comportamiento especifico

Prompt_template = PromptTemplate(
    input_variables=["mensaje","historial"],
    template="""Eres un asistente util y amigable llamdo Chatbot Pro.

Historial de conversación:
{historial}
Responde de manera clara y concisa a la siguiente pregunta: {mensaje}"""
)

# Crear cadena usando LCEL (Langchain Expression Language)
cadena = Prompt_template | chat_model


# Mostrar mensajes previos en la interfaz 
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        #No muestro el mensaje por pantalla 
        continue
    
    role = "asistant" if isinstance(msg, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(msg.content)

if st.button("🗑️Nueva conversación"):
    st.session_state.mensajes = []
    st.rerun()

# Cuadro de entrada de texto de usuario 

pregunta = st.chat_input("Escribe tu mensaje: ")

if pregunta:
    # Mostrar inmediatamente el mensaje del usuario en la interfaz 
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Almacenamos el mensaje en la memoria de streamlit 
    st.session_state.mensajes.append(HumanMessage(content=pregunta))

    # Generar respuesta usando el modelo de lenguaje 
    respuesta = chat_model.invoke(st.session_state.mensajes)

    # Mostrar la respuesta en la interfaz 

    with st.chat_message("assistant"):
        st.markdown(respuesta.content)

    st.session_state.mensajes.append(respuesta)
