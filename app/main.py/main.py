import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from app.ui.sidebar import render_sidebar
from app.ui.layout import render_historial
from app.core.prompt_factory import crear_prompt
from app.core.chat_engine import crear_modelo
from app.core.memory import LongTermMemory, historial_a_texto
from app.core.rag_engine import RAGEngine


# -----------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(page_title="Chatbot Pro", page_icon="🤖")
st.title("🤖 Chatbot Pro con Memoria + RAG")


# -----------------------------
# ESTADO INICIAL
# -----------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


# -----------------------------
# SIDEBAR (temperatura, modelo, PDF)
# -----------------------------
temperature, model_name, pdf_file = render_sidebar()


# -----------------------------
# INICIALIZAR MOTORES
# -----------------------------
memory = LongTermMemory()
rag = RAGEngine()


# -----------------------------
# CARGA DE PDF (si el usuario sube uno)
# -----------------------------
if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    paginas = rag.add_pdf("temp.pdf")
    st.sidebar.success(f"PDF cargado. Páginas indexadas: {paginas}")


# -----------------------------
# ENTRADA DEL USUARIO
# -----------------------------
pregunta = st.chat_input("Escribe tu mensaje...")

if pregunta:
    # Guardar en memoria a largo plazo
    memory.save_memory(pregunta)

    # Recuperar historial corto
    historial_text = historial_a_texto(st.session_state.mensajes)

    # Recuperar memoria relevante
    recuerdos = memory.retrieve_memory(pregunta)
    contexto_memoria = "\n".join(recuerdos)

    # Recuperar documentos relevantes (RAG)
    resultados_rag = rag.search(pregunta)
    contexto_rag = "\n".join(resultados_rag)

    # Crear prompt
    prompt = crear_prompt()

    # Crear modelo
    modelo = crear_modelo(model_name, temperature)

    # Ejecutar cadena
    cadena = prompt | modelo

    st.chat_message("user").markdown(pregunta)
    response_placeholder = st.chat_message("assistant")

    full_response = ""

    try:
        for chunk in cadena.stream({
            "mensaje": pregunta,
            "historial": historial_text,
            "memoria": contexto_memoria,
            "documentos": contexto_rag
        }):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # Guardar en historial
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        st.session_state.mensajes.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica tu API Key de OpenAI.")


# -----------------------------
# MOSTRAR HISTORIAL
# -----------------------------
render_historial(st.session_state.mensajes)
