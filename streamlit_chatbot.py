import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from limite_mensajes import LimitadorMensajes

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
st.set_page_config(page_title="Chatbot con LangChain", page_icon="🤖")


def leer_config(nombre: str, por_defecto=None):
    """Lee un valor de .streamlit/secrets.toml (o de los Secrets de
    Streamlit Cloud) y, si no está, de las variables de entorno."""
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:
        pass  # No existe secrets.toml: seguimos con variables de entorno
    return os.environ.get(nombre, por_defecto)


MAX_MENSAJES_POR_USUARIO = int(leer_config("MAX_MENSAJES_POR_USUARIO", 20))
MAX_MENSAJES_GLOBALES_DIA = int(leer_config("MAX_MENSAJES_GLOBALES_DIA", 300))
MAX_TOKENS_RESPUESTA = int(leer_config("MAX_TOKENS_RESPUESTA", 1000))
MAX_CARACTERES_PREGUNTA = 2000
MENSAJES_DE_HISTORIAL = 10  # cuántos mensajes previos recibe el modelo

MODELOS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini"]

PERSONALIDADES = {
    "Útil y amigable": "Eres un asistente útil y amigable llamado ChatBot Pro. Responde de manera clara y concisa.",
    "Profesional y formal": "Eres un asistente profesional y formal. Proporciona respuestas precisas y bien estructuradas.",
    "Casual y relajado": "Eres un asistente casual y relajado. Habla de forma natural y amigable, como un buen amigo.",
    "Experto técnico": "Eres un asistente experto técnico. Proporciona respuestas detalladas con precisión técnica.",
    "Creativo y divertido": "Eres un asistente creativo y divertido. Usa analogías, ejemplos creativos y mantén un tono alegre.",
}


@st.cache_resource
def obtener_limitador() -> LimitadorMensajes:
    # Una sola instancia compartida por todas las sesiones de la app
    return LimitadorMensajes(MAX_MENSAJES_POR_USUARIO, MAX_MENSAJES_GLOBALES_DIA)


def identificar_usuario() -> str:
    """Identifica al visitante por su IP; si no se puede, por su sesión."""
    headers = st.context.headers or {}
    reenviada = headers.get("X-Forwarded-For")
    if reenviada:
        return "ip:" + reenviada.split(",")[0].strip()
    ip = st.context.ip_address
    if isinstance(ip, str) and ip:
        return "ip:" + ip
    if "id_sesion" not in st.session_state:
        st.session_state.id_sesion = str(uuid.uuid4())
    return "sesion:" + st.session_state.id_sesion


def crear_modelo(modelo: str, temperatura: float) -> ChatOpenAI:
    opciones = {"model": modelo, "streaming": True}
    api_key = leer_config("OPENAI_API_KEY")
    if api_key:
        opciones["api_key"] = api_key
    if modelo.startswith("gpt-5"):
        # Los modelos de razonamiento no aceptan temperatura y necesitan más
        # tokens porque parte se usa para "pensar" antes de responder.
        opciones["reasoning_effort"] = "low"
        opciones["max_tokens"] = MAX_TOKENS_RESPUESTA * 4
    else:
        opciones["temperature"] = temperatura
        opciones["max_tokens"] = MAX_TOKENS_RESPUESTA
    return ChatOpenAI(**opciones)


limitador = obtener_limitador()
usuario = identificar_usuario()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# -----------------------------
# BARRA LATERAL
# -----------------------------
with st.sidebar:
    st.header("Configuración")
    model_name = st.selectbox("Modelo", MODELOS)
    es_razonamiento = model_name.startswith("gpt-5")
    temperature = st.slider(
        "Temperatura",
        0.0,
        1.0,
        0.5,
        0.1,
        disabled=es_razonamiento,
        help="Los modelos GPT-5 no permiten ajustar la temperatura." if es_razonamiento else None,
    )
    personalidad = st.selectbox("Personalidad del asistente", list(PERSONALIDADES))

    st.divider()
    restantes = limitador.restantes(usuario)
    st.caption(f"💬 Te quedan **{restantes} de {MAX_MENSAJES_POR_USUARIO}** mensajes hoy.")
    st.progress(restantes / MAX_MENSAJES_POR_USUARIO if MAX_MENSAJES_POR_USUARIO else 0)

    if st.button("🗑️ Nueva conversación", width="stretch"):
        st.session_state.mensajes = []
        st.rerun()

# -----------------------------
# CADENA (LCEL)
# -----------------------------
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instrucciones}"),
        MessagesPlaceholder("historial"),
        ("human", "{mensaje}"),
    ]
)
cadena = chat_prompt | crear_modelo(model_name, temperature)

# -----------------------------
# CHAT
# -----------------------------
st.title("🤖 Chatbot con LangChain")
st.markdown(
    "Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. "
    "¡Escribe tu mensaje abajo para comenzar!"
)

for msg in st.session_state.mensajes:
    rol = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(rol):
        st.markdown(msg.content)

pregunta = st.chat_input("Escribe tu mensaje:", max_chars=MAX_CARACTERES_PREGUNTA)

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)

    rechazo = limitador.intentar_consumir(usuario)
    if rechazo == "usuario":
        st.warning(
            f"Llegaste al límite de {MAX_MENSAJES_POR_USUARIO} mensajes por hoy. "
            "Vuelve mañana para seguir conversando. 🙌"
        )
        st.stop()
    if rechazo == "global":
        st.warning(
            "El chatbot alcanzó su límite de uso diario. Vuelve a intentarlo mañana. 🙏"
        )
        st.stop()

    historial = st.session_state.mensajes[-MENSAJES_DE_HISTORIAL:]
    # Se guarda la pregunta antes de responder para que no se pierda si hay un error
    st.session_state.mensajes.append(HumanMessage(content=pregunta))

    with st.chat_message("assistant"):
        try:
            respuesta = st.write_stream(
                chunk.content
                for chunk in cadena.stream(
                    {
                        "instrucciones": PERSONALIDADES[personalidad],
                        "historial": historial,
                        "mensaje": pregunta,
                    }
                )
            )
        except Exception as e:
            limitador.devolver(usuario)  # el mensaje no se cobra si OpenAI falló
            st.error(f"Error al generar respuesta: {e}")
            st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")
        else:
            st.session_state.mensajes.append(AIMessage(content=respuesta))
            st.rerun()  # actualiza el contador de mensajes de la barra lateral
