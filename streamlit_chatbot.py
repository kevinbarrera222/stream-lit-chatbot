import itertools
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from adjuntos import (
    ErrorAdjunto,
    a_markdown,
    a_mensaje_langchain,
    boton_escuchar_html,
    procesar_imagen,
    procesar_pdf,
    transcribir_audio,
)
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
MAX_ARCHIVOS_POR_MENSAJE = 3
MAX_MB_POR_ARCHIVO = 10
MAX_MB_AUDIO = 4  # ≈ 2 minutos de voz
TIPOS_IMAGEN = ["png", "jpg", "jpeg", "webp", "gif"]

# GPT-6 Luna: el modelo más económico de OpenAI, lee imágenes y reemplaza a
# gpt-4o-mini / gpt-4.1-mini (que OpenAI retira de su API el 14 de enero de 2027).
# Se pueden ofrecer otros modelos en Secrets: MODELOS = ["gpt-6-luna", "gpt-6-sol"]
_modelos = leer_config("MODELOS", "gpt-6-luna")
if isinstance(_modelos, str):
    _modelos = _modelos.split(",")
MODELOS = [str(m).strip() for m in _modelos if str(m).strip()] or ["gpt-6-luna"]
MODELO_TRANSCRIPCION = leer_config("MODELO_TRANSCRIPCION", "gpt-transcribe")

MODOS_RESPUESTA = {  # nombre visible → esfuerzo de razonamiento
    "⚡ Rápido": "none",
    "⚖️ Equilibrado": "low",
    "🧠 Profundo": "medium",
}

INSTRUCCIONES_GENERALES = (
    " Responde en el mismo idioma en que te escriben. Si el usuario adjunta imágenes o PDFs, "
    "úsalos para responder."
)
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


def es_modelo_de_razonamiento(modelo: str) -> bool:
    return modelo.startswith(("gpt-5", "gpt-6", "o"))


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


def crear_modelo(modelo: str, temperatura: float, esfuerzo: str) -> ChatOpenAI:
    opciones = {"model": modelo, "streaming": True}
    api_key = leer_config("OPENAI_API_KEY")
    if api_key:
        opciones["api_key"] = api_key
    if es_modelo_de_razonamiento(modelo):
        # Los modelos de razonamiento no usan temperatura. Si "piensan", parte de
        # los tokens se va en el razonamiento, por eso se les da más margen.
        opciones["reasoning_effort"] = esfuerzo
        opciones["max_tokens"] = MAX_TOKENS_RESPUESTA * (1 if esfuerzo == "none" else 4)
    else:
        opciones["temperature"] = temperatura
        opciones["max_tokens"] = MAX_TOKENS_RESPUESTA
    return ChatOpenAI(**opciones)


def conversacion_a_markdown(mensajes) -> str:
    fecha = datetime.now(ZONA_HORARIA).strftime("%Y-%m-%d %H:%M")
    partes = [f"# Conversación con {NOMBRE_BOT}\n\n_{fecha}_\n"]
    for msg in mensajes:
        autor = NOMBRE_BOT if msg["rol"] == "assistant" else "Tú"
        partes.append(f"**{autor}:**\n\n{a_markdown(msg)}\n")
    return "\n---\n\n".join(partes)


def mostrar_mensaje_usuario(msg: dict) -> None:
    with st.chat_message("user", avatar=AVATAR_USUARIO):
        if msg.get("voz"):
            st.caption(":material/mic: Mensaje de voz")
        if msg["texto"]:
            st.markdown(msg["texto"])
        if msg.get("imagenes"):
            with st.container(horizontal=True, gap="small"):
                for imagen in msg["imagenes"]:
                    st.image(imagen, width=220)
        for pdf in msg.get("pdfs") or []:
            paginas = "1 página" if pdf["paginas"] == 1 else f"{pdf['paginas']} páginas"
            st.caption(f":material/picture_as_pdf: **{pdf['nombre']}** · {paginas}")


def preparar_adjuntos(archivos) -> tuple[list, list]:
    """Convierte los archivos subidos en imágenes reducidas y texto de PDFs."""
    if len(archivos) > MAX_ARCHIVOS_POR_MENSAJE:
        raise ErrorAdjunto(f"Puedes adjuntar hasta {MAX_ARCHIVOS_POR_MENSAJE} archivos por mensaje.")
    imagenes, pdfs = [], []
    for archivo in archivos:
        extension = archivo.name.rsplit(".", 1)[-1].lower()
        if extension == "pdf":
            pdfs.append(procesar_pdf(archivo.name, archivo.getvalue()))
        elif extension in TIPOS_IMAGEN:
            imagenes.append(procesar_imagen(archivo.getvalue()))
        else:
            raise ErrorAdjunto(f"No puedo leer archivos .{extension}. Usa imágenes o PDFs.")
    return imagenes, pdfs


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
    st.session_state.pendiente = {"rol": "user", "texto": texto}


def regenerar() -> None:
    """Quita la última respuesta y vuelve a enviar la última pregunta (con sus adjuntos)."""
    mensajes = st.session_state.mensajes
    if len(mensajes) >= 2 and mensajes[-1]["rol"] == "assistant":
        st.session_state.pop(f"valoracion_{len(mensajes) - 1}", None)
        mensajes.pop()  # respuesta anterior
        # Un mensaje de voz ya está transcrito: no se vuelve a transcribir
        st.session_state.pendiente = mensajes.pop()


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
    if len(MODELOS) > 1:
        model_name = st.selectbox("Modelo", MODELOS, key="modelo")
    else:
        model_name = st.session_state["modelo"] = MODELOS[0]
        st.caption(f":material/memory: Modelo: **{model_name}**")

    temperature, esfuerzo = 0.5, "none"
    if es_modelo_de_razonamiento(model_name):
        modo = st.selectbox(
            "Modo de respuesta",
            list(MODOS_RESPUESTA),
            help="Rápido responde al instante. Equilibrado y Profundo piensan antes de "
            "responder: mejores para problemas difíciles, pero tardan más.",
        )
        esfuerzo = MODOS_RESPUESTA[modo]
    else:
        temperature = st.slider(
            "Temperatura", 0.0, 1.0, 0.5, 0.1,
            help="Más baja = respuestas más precisas. Más alta = más creativas.",
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
    st.caption(
        ":material/attach_file: Puedes adjuntar imágenes y PDFs, o "
        ":material/mic: grabar un mensaje de voz desde la caja de texto."
    )

# -----------------------------
# CADENA (LCEL)
# -----------------------------
API_KEY = leer_config("OPENAI_API_KEY")
if not API_KEY:
    st.error(
        "🔑 Falta la API Key de OpenAI. Agrega `OPENAI_API_KEY` en los Secrets de la app "
        "(en Streamlit Cloud: Manage app → ⋮ → Settings → Secrets) o en `.streamlit/secrets.toml`."
    )
    st.stop()

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instrucciones}"),
        MessagesPlaceholder("historial"),
        MessagesPlaceholder("pregunta"),
    ]
)
cadena = chat_prompt | crear_modelo(model_name, temperature, esfuerzo)

# -----------------------------
# PANTALLA DE BIENVENIDA
# -----------------------------
if not st.session_state.mensajes and not st.session_state.pendiente:
    st.markdown(f"## ¡Hola! Soy {NOMBRE_BOT} 👋")
    st.markdown(
        "Pregúntame lo que quieras: ideas, textos, explicaciones o planes. También puedo "
        "**leer imágenes y PDFs** y **escuchar mensajes de voz**. Para empezar, prueba con una de estas:"
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
    if msg["rol"] == "user":
        mostrar_mensaje_usuario(msg)
        continue
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        st.markdown(msg["texto"])
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.feedback(
                "thumbs",
                key=f"valoracion_{i}",
                on_change=registrar_valoracion,
                args=(i,),
            )
            with st.popover("Copiar", icon=":material/content_copy:", type="tertiary"):
                st.caption("Usa el ícono de copiar de la esquina:")
                st.code(msg["texto"], language="markdown", wrap_lines=True)
            st.html(
                boton_escuchar_html(f"{i}-{len(msg['texto'])}", msg["texto"]),
                unsafe_allow_javascript=True,
                width="content",
            )
            if i == ultimo:
                st.button(
                    "Regenerar",
                    icon=":material/refresh:",
                    type="tertiary",
                    on_click=regenerar,
                    key="regenerar",
                )

# -----------------------------
# NUEVA PREGUNTA
# -----------------------------
entrada = st.chat_input(
    f"Escribe tu mensaje a {NOMBRE_BOT}…",
    max_chars=MAX_CARACTERES_PREGUNTA,
    accept_file="multiple",
    file_type=TIPOS_IMAGEN + ["pdf"],
    max_upload_size=MAX_MB_POR_ARCHIVO,
    accept_audio=True,
)

pregunta = None  # mensaje del usuario ya listo (diccionario)
audio = None
if st.session_state.pendiente:
    pregunta, st.session_state.pendiente = st.session_state.pendiente, None
elif entrada:
    texto = (entrada.text or "").strip()
    archivos = list(entrada.get("files") or [])
    audio = entrada.get("audio")
    try:
        imagenes, pdfs = preparar_adjuntos(archivos)
        if audio is not None and audio.size > MAX_MB_AUDIO * 1024 * 1024:
            raise ErrorAdjunto("El audio es muy largo. Graba mensajes de máximo unos 2 minutos.")
    except ErrorAdjunto as e:
        st.warning(str(e))
        st.stop()
    if not texto and audio is None:
        if pdfs:
            texto = "Resume este documento." if len(pdfs) == 1 else "Resume estos documentos."
        elif imagenes:
            texto = "Describe esta imagen." if len(imagenes) == 1 else "Describe estas imágenes."
    if texto or audio is not None:
        pregunta = {"rol": "user", "texto": texto, "imagenes": imagenes, "pdfs": pdfs, "voz": audio is not None}

if pregunta:
    rechazo = limitador.intentar_consumir(usuario)
    if rechazo:
        mostrar_mensaje_usuario(pregunta)
        if rechazo == "usuario":
            st.warning(
                f"Llegaste al límite de {MAX_MENSAJES_POR_USUARIO} mensajes por hoy. "
                "Vuelve mañana para seguir conversando. 🙌"
            )
        else:
            st.warning("El chatbot alcanzó su límite de uso diario. Vuelve a intentarlo mañana. 🙏")
        st.stop()

    # Mensaje de voz: primero se transcribe (cuenta como el mismo mensaje)
    if audio is not None:
        try:
            with st.spinner("Escuchando tu mensaje…"):
                transcripcion = transcribir_audio(audio.getvalue(), API_KEY, MODELO_TRANSCRIPCION)
        except Exception as e:
            limitador.devolver(usuario)
            st.error(f"No pude transcribir el audio: {e}")
            st.stop()
        if not transcripcion:
            limitador.devolver(usuario)
            st.warning("No logré entender el audio. ¿Puedes intentarlo de nuevo, más cerca del micrófono?")
            st.stop()
        pregunta["texto"] = (pregunta["texto"] + "\n\n" + transcripcion).strip()

    mostrar_mensaje_usuario(pregunta)
    historial = [a_mensaje_langchain(m) for m in st.session_state.mensajes[-MENSAJES_DE_HISTORIAL:]]
    # Se guarda la pregunta antes de responder para que no se pierda si hay un error
    st.session_state.mensajes.append(pregunta)

    with st.chat_message("assistant", avatar=AVATAR_BOT):
        try:
            fragmentos = (
                chunk.content
                for chunk in cadena.stream(
                    {
                        "instrucciones": PERSONALIDADES[personalidad] + INSTRUCCIONES_GENERALES,
                        "historial": historial,
                        "pregunta": [a_mensaje_langchain(pregunta)],
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
            st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta})
            st.rerun()  # muestra los botones de la respuesta y actualiza el contador
