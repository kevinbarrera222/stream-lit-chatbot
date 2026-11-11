import itertools
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from limite_mensajes import ZONA_HORARIA, LimitadorMensajes

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
NOMBRE_BOT = "ChatBot Pro"
CARPETA_ASSETS = Path(__file__).parent / "assets"
LOGO = str(CARPETA_ASSETS / "logo.svg")
ICONO = str(CARPETA_ASSETS / "icono.svg")
AVATAR_BOT = ICONO
AVATAR_USUARIO = ":material/person:"

st.set_page_config(page_title=NOMBRE_BOT, page_icon=ICONO, layout="centered")


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

SUGERENCIAS = [  # (ícono, texto del botón, pregunta que se envía)
    (":material/lightbulb:", "Ideas de negocio con IA", "Dame 3 ideas de negocio que se puedan empezar con IA"),
    (":material/mail:", "Escribir un correo profesional", "Ayúdame a escribir un correo profesional para pedir una reunión"),
    (":material/school:", "Explicar la IA fácil", "Explícame qué es la inteligencia artificial como si tuviera 10 años"),
    (":material/flight:", "Viaje con poco presupuesto", "Planea un fin de semana de viaje con poco presupuesto"),
]


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


def conversacion_a_markdown(mensajes) -> str:
    fecha = datetime.now(ZONA_HORARIA).strftime("%Y-%m-%d %H:%M")
    partes = [f"# Conversación con {NOMBRE_BOT}\n\n_{fecha}_\n"]
    for msg in mensajes:
        autor = NOMBRE_BOT if isinstance(msg, AIMessage) else "Tú"
        partes.append(f"**{autor}:**\n\n{msg.content}\n")
    return "\n---\n\n".join(partes)


# -----------------------------
# ACCIONES (callbacks de botones)
# -----------------------------
def limpiar_valoraciones() -> None:
    for clave in [c for c in st.session_state if str(c).startswith("valoracion_")]:
        del st.session_state[clave]


def nueva_conversacion() -> None:
    st.session_state.mensajes = []
    st.session_state.pendiente = None
    limpiar_valoraciones()


def usar_sugerencia(texto: str) -> None:
    st.session_state.pendiente = texto


def regenerar() -> None:
    """Quita la última respuesta y vuelve a enviar la última pregunta."""
    mensajes = st.session_state.mensajes
    if len(mensajes) >= 2 and isinstance(mensajes[-1], AIMessage):
        indice = len(mensajes) - 1
        st.session_state.pop(f"valoracion_{indice}", None)
        mensajes.pop()  # respuesta anterior
        st.session_state.pendiente = mensajes.pop().content  # pregunta


def registrar_valoracion(indice: int) -> None:
    valor = st.session_state.get(f"valoracion_{indice}")
    etiqueta = {0: "👎", 1: "👍"}.get(valor, "sin valoración")
    # Aparece en los logs de Streamlit Cloud (Manage app → logs)
    print(f"[valoracion] respuesta #{indice}: {etiqueta} | modelo={st.session_state.get('modelo')}")


# -----------------------------
# ESTADO
# -----------------------------
limitador = obtener_limitador()
usuario = identificar_usuario()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "pendiente" not in st.session_state:
    st.session_state.pendiente = None

st.logo(LOGO, icon_image=ICONO, size="large")

# -----------------------------
# BARRA LATERAL
# -----------------------------
with st.sidebar:
    st.subheader("Configuración")
    model_name = st.selectbox("Modelo", MODELOS, key="modelo")
    es_razonamiento = model_name.startswith("gpt-5")
    temperature = st.slider(
        "Temperatura",
        0.0,
        1.0,
        0.5,
        0.1,
        disabled=es_razonamiento,
        help=(
            "Los modelos GPT-5 no permiten ajustar la temperatura."
            if es_razonamiento
            else "Más baja = respuestas más precisas. Más alta = más creativas."
        ),
    )
    personalidad = st.selectbox("Personalidad del asistente", list(PERSONALIDADES))

    st.divider()
    restantes = limitador.restantes(usuario)
    st.caption(f":material/chat: Te quedan **{restantes} de {MAX_MENSAJES_POR_USUARIO}** mensajes hoy.")
    st.progress(restantes / MAX_MENSAJES_POR_USUARIO if MAX_MENSAJES_POR_USUARIO else 0)

    hay_mensajes = bool(st.session_state.mensajes)
    st.button(
        "Nueva conversación",
        icon=":material/add_comment:",
        on_click=nueva_conversacion,
        width="stretch",
        type="primary",
    )
    st.download_button(
        "Descargar conversación",
        data=conversacion_a_markdown(st.session_state.mensajes) if hay_mensajes else "",
        file_name=f"conversacion-{datetime.now(ZONA_HORARIA):%Y%m%d-%H%M}.md",
        mime="text/markdown",
        icon=":material/download:",
        width="stretch",
        disabled=not hay_mensajes,
    )

# -----------------------------
# CADENA (LCEL)
# -----------------------------
if not leer_config("OPENAI_API_KEY"):
    st.error(
        "🔑 Falta la API Key de OpenAI. Agrega `OPENAI_API_KEY` en los Secrets de la app "
        "(en Streamlit Cloud: Manage app → ⋮ → Settings → Secrets) o en `.streamlit/secrets.toml`."
    )
    st.stop()

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instrucciones}"),
        MessagesPlaceholder("historial"),
        ("human", "{mensaje}"),
    ]
)
cadena = chat_prompt | crear_modelo(model_name, temperature)

# -----------------------------
# PANTALLA DE BIENVENIDA
# -----------------------------
if not st.session_state.mensajes and not st.session_state.pendiente:
    st.markdown(f"## ¡Hola! Soy {NOMBRE_BOT} 👋")
    st.markdown(
        "Pregúntame lo que quieras: ideas, textos, explicaciones o planes. "
        "Puedes cambiar mi personalidad en la barra lateral. Para empezar, prueba con una de estas:"
    )
    columnas = st.columns(2)
    for i, (icono, etiqueta, texto) in enumerate(SUGERENCIAS):
        columnas[i % 2].button(
            etiqueta,
            icon=icono,
            key=f"sugerencia_{i}",
            on_click=usar_sugerencia,
            args=(texto,),
            width="stretch",
        )

# -----------------------------
# HISTORIAL
# -----------------------------
ultimo = len(st.session_state.mensajes) - 1
for i, msg in enumerate(st.session_state.mensajes):
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=AVATAR_BOT):
            st.markdown(msg.content)
            with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                st.feedback(
                    "thumbs",
                    key=f"valoracion_{i}",
                    on_change=registrar_valoracion,
                    args=(i,),
                )
                with st.popover("Copiar", icon=":material/content_copy:", type="tertiary"):
                    st.caption("Usa el ícono de copiar de la esquina:")
                    st.code(msg.content, language="markdown", wrap_lines=True)
                if i == ultimo:
                    st.button(
                        "Regenerar",
                        icon=":material/refresh:",
                        type="tertiary",
                        on_click=regenerar,
                        key="regenerar",
                    )
    else:
        with st.chat_message("user", avatar=AVATAR_USUARIO):
            st.markdown(msg.content)

# -----------------------------
# NUEVA PREGUNTA
# -----------------------------
pregunta = st.chat_input(f"Escribe tu mensaje a {NOMBRE_BOT}…", max_chars=MAX_CARACTERES_PREGUNTA)
if st.session_state.pendiente:
    pregunta = st.session_state.pendiente
    st.session_state.pendiente = None

if pregunta:
    with st.chat_message("user", avatar=AVATAR_USUARIO):
        st.markdown(pregunta)

    rechazo = limitador.intentar_consumir(usuario)
    if rechazo == "usuario":
        st.warning(
            f"Llegaste al límite de {MAX_MENSAJES_POR_USUARIO} mensajes por hoy. "
            "Vuelve mañana para seguir conversando. 🙌"
        )
        st.stop()
    if rechazo == "global":
        st.warning("El chatbot alcanzó su límite de uso diario. Vuelve a intentarlo mañana. 🙏")
        st.stop()

    historial = st.session_state.mensajes[-MENSAJES_DE_HISTORIAL:]
    # Se guarda la pregunta antes de responder para que no se pierda si hay un error
    st.session_state.mensajes.append(HumanMessage(content=pregunta))

    with st.chat_message("assistant", avatar=AVATAR_BOT):
        try:
            fragmentos = (
                chunk.content
                for chunk in cadena.stream(
                    {
                        "instrucciones": PERSONALIDADES[personalidad],
                        "historial": historial,
                        "mensaje": pregunta,
                    }
                )
            )
            # "Pensando…" se muestra hasta que llega la primera palabra
            with st.spinner("Pensando…"):
                primero = next(fragmentos, "")
            respuesta = st.write_stream(itertools.chain([primero], fragmentos))
        except Exception as e:
            limitador.devolver(usuario)  # el mensaje no se cobra si OpenAI falló
            st.error(f"Error al generar respuesta: {e}")
            st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")
        else:
            st.session_state.mensajes.append(AIMessage(content=respuesta))
            st.rerun()  # muestra los botones de la respuesta y actualiza el contador
