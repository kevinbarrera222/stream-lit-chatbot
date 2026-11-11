"""Funciones para imágenes, PDFs y voz.

Cada mensaje de la conversación se guarda como un diccionario:

    {
        "rol": "user" | "assistant",
        "texto": str,
        "imagenes": [bytes JPEG],                       # solo usuario
        "pdfs": [{"nombre", "paginas", "texto", "recortado"}],  # solo usuario
        "voz": bool,                                    # True si llegó por audio
    }
"""

from __future__ import annotations

import base64
import io
import re

from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image
from pypdf import PdfReader

LADO_MAXIMO_IMAGEN = 1568       # px; más grande no mejora la lectura y cuesta más
MAX_CARACTERES_PDF = 15000      # texto que se envía al modelo por cada PDF
MAX_PAGINAS_PDF = 50


class ErrorAdjunto(Exception):
    """Error con un mensaje listo para mostrar al usuario."""


# -----------------------------
# IMÁGENES
# -----------------------------
def procesar_imagen(datos: bytes) -> bytes:
    """Reduce la imagen y la convierte a JPEG para ahorrar tokens."""
    try:
        imagen = Image.open(io.BytesIO(datos))
        imagen.seek(0)  # en GIF animados se usa el primer cuadro
        imagen = imagen.convert("RGB")
    except Exception as e:
        raise ErrorAdjunto("No pude abrir la imagen. ¿Está dañada?") from e
    imagen.thumbnail((LADO_MAXIMO_IMAGEN, LADO_MAXIMO_IMAGEN))
    salida = io.BytesIO()
    imagen.save(salida, format="JPEG", quality=85)
    return salida.getvalue()


# -----------------------------
# PDFs
# -----------------------------
def procesar_pdf(nombre: str, datos: bytes) -> dict:
    """Extrae el texto del PDF (hasta MAX_PAGINAS_PDF y MAX_CARACTERES_PDF)."""
    try:
        lector = PdfReader(io.BytesIO(datos))
        if lector.is_encrypted:
            lector.decrypt("")
        paginas = len(lector.pages)
        partes = []
        for pagina in lector.pages[:MAX_PAGINAS_PDF]:
            partes.append(pagina.extract_text() or "")
    except Exception as e:
        raise ErrorAdjunto(f"No pude leer el PDF «{nombre}». ¿Está dañado o protegido con contraseña?") from e

    texto = re.sub(r"[ \t]+", " ", "\n".join(partes)).strip()
    if not texto:
        raise ErrorAdjunto(
            f"El PDF «{nombre}» no tiene texto que se pueda leer (parece escaneado). "
            "Envía una foto o captura de la página como imagen."
        )
    recortado = paginas > MAX_PAGINAS_PDF or len(texto) > MAX_CARACTERES_PDF
    return {
        "nombre": nombre,
        "paginas": paginas,
        "texto": texto[:MAX_CARACTERES_PDF],
        "recortado": recortado,
    }


# -----------------------------
# CONVERSIÓN PARA EL MODELO
# -----------------------------
def a_mensaje_langchain(mensaje: dict):
    if mensaje["rol"] == "assistant":
        return AIMessage(content=mensaje["texto"])

    imagenes = mensaje.get("imagenes") or []
    pdfs = mensaje.get("pdfs") or []
    if not imagenes and not pdfs:
        return HumanMessage(content=mensaje["texto"])

    texto = mensaje["texto"]
    for pdf in pdfs:
        aviso = " (solo una parte del documento)" if pdf["recortado"] else ""
        texto += (
            f"\n\n--- Contenido del PDF «{pdf['nombre']}», {pdf['paginas']} páginas{aviso} ---\n"
            f"{pdf['texto']}\n--- Fin del PDF ---"
        )
    contenido = [{"type": "text", "text": texto}]
    for imagen in imagenes:
        url = "data:image/jpeg;base64," + base64.b64encode(imagen).decode()
        contenido.append({"type": "image_url", "image_url": {"url": url}})
    return HumanMessage(content=contenido)


def a_markdown(mensaje: dict) -> str:
    """Texto del mensaje para la conversación descargada."""
    extras = []
    if mensaje.get("voz"):
        extras.append("🎤 _Mensaje de voz_")
    extras += ["🖼️ _Imagen adjunta_"] * len(mensaje.get("imagenes") or [])
    extras += [f"📄 _PDF: {p['nombre']}_" for p in mensaje.get("pdfs") or []]
    return "\n\n".join(extras + [mensaje["texto"]])


# -----------------------------
# VOZ
# -----------------------------
def transcribir_audio(datos: bytes, api_key: str | None, modelo: str) -> str:
    """Convierte el audio grabado en texto con la API de OpenAI."""
    from openai import OpenAI

    cliente = OpenAI(api_key=api_key) if api_key else OpenAI()
    archivo = ("mensaje.wav", datos, "audio/wav")
    resultado = cliente.audio.transcriptions.create(model=modelo, file=archivo)
    return (resultado.text or "").strip()


def limpiar_para_voz(texto: str) -> str:
    """Quita el formato Markdown para que la voz no lea asteriscos ni símbolos."""
    texto = re.sub(r"```.*?```", " (bloque de código) ", texto, flags=re.S)
    texto = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", texto)        # imágenes
    texto = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", texto)     # enlaces → solo el texto
    texto = re.sub(r"`([^`]*)`", r"\1", texto)
    texto = re.sub(r"^\s{0,3}#{1,6}\s*", "", texto, flags=re.M)  # títulos
    texto = re.sub(r"^\s*[-*+]\s+", "", texto, flags=re.M)        # viñetas
    texto = re.sub(r"^\s*>\s?", "", texto, flags=re.M)            # citas
    texto = re.sub(r"(\*\*|__|\*|_|~~)", "", texto)
    texto = re.sub(r"^\s*([-*_]\s*){3,}$", "", texto, flags=re.M)  # separadores
    texto = re.sub(r"\|", " ", texto)
    return re.sub(r"\n{2,}", "\n", texto).strip()


def boton_escuchar_html(clave: str, texto: str) -> str:
    """Botón que lee la respuesta en voz alta con la voz del navegador.

    Usa la síntesis de voz del propio navegador (Web Speech API): es gratis,
    no gasta saldo de OpenAI y no depende de ningún modelo que se pueda retirar.
    """
    # El texto va en base64 para que ningún carácter de la respuesta (<, comillas, etc.)
    # pueda romper el HTML o el JavaScript.
    texto_b64 = base64.b64encode(limpiar_para_voz(texto).encode("utf-8")).decode("ascii")
    id_boton = "voz-" + re.sub(r"[^a-zA-Z0-9_-]", "", clave)
    return f"""
<style>
  .cb-voz {{ display:inline-flex; align-items:center; gap:.35rem; background:none; border:none;
            color:inherit; font:inherit; font-size:.875rem; cursor:pointer; padding:.25rem .1rem; }}
  .cb-voz:hover {{ color:#A78BFA; }}
  .cb-voz .cb-icono {{ font-family:'Material Symbols Rounded'; font-size:1.15rem; line-height:1;
                      font-weight:normal; font-style:normal; }}
</style>
<button class="cb-voz" id="{id_boton}" type="button" title="Leer la respuesta en voz alta">
  <span class="cb-icono">volume_up</span><span class="cb-texto">Escuchar</span>
</button>
<script>
(() => {{
  const boton = document.getElementById("{id_boton}");
  if (!boton || boton.dataset.listo) return;
  boton.dataset.listo = "1";
  const bytes = Uint8Array.from(atob("{texto_b64}"), c => c.charCodeAt(0));
  const texto = new TextDecoder().decode(bytes);
  const estado = (b, hablando) => {{
    b.dataset.hablando = hablando ? "1" : "";
    b.querySelector(".cb-icono").textContent = hablando ? "stop_circle" : "volume_up";
    b.querySelector(".cb-texto").textContent = hablando ? "Detener" : "Escuchar";
  }};
  const elegirVoz = () => {{
    const voces = window.speechSynthesis.getVoices();
    for (const pref of ["es-us", "es-mx", "es-419", "es-co", "es-es"]) {{
      const v = voces.find(v => (v.lang || "").replace("_", "-").toLowerCase() === pref);
      if (v) return v;
    }}
    return voces.find(v => (v.lang || "").toLowerCase().startsWith("es")) || null;
  }};
  boton.addEventListener("click", () => {{
    const sintesis = window.speechSynthesis;
    if (!sintesis) {{ boton.querySelector(".cb-texto").textContent = "Voz no disponible"; return; }}
    const estabaHablando = boton.dataset.hablando === "1";
    sintesis.cancel();
    document.querySelectorAll(".cb-voz").forEach(b => estado(b, false));
    if (estabaHablando) return;
    // Se lee por frases: Chrome corta los textos largos leídos de una sola vez
    const partes = (texto.match(/[^.!?\\n]+[.!?]*/g) || [texto]).map(f => f.trim()).filter(Boolean);
    const voz = elegirVoz();
    partes.forEach((frase, i) => {{
      const u = new SpeechSynthesisUtterance(frase);
      if (voz) {{ u.voice = voz; u.lang = voz.lang; }} else {{ u.lang = "es-US"; }}
      if (i === partes.length - 1) {{ u.onend = () => estado(boton, false); u.onerror = u.onend; }}
      sintesis.speak(u);
    }});
    estado(boton, true);
  }});
}})();
</script>
"""
