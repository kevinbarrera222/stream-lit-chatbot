## 🤖 Chatbot con Streamlit + LangChain

Un chatbot hecho con **Streamlit**, **LangChain** y **OpenAI**, con respuestas en tiempo real, personalidades configurables, historial de conversación y **límite de mensajes por usuario** para controlar el gasto de la API.

---

## 🚀 Funcionalidades

- 🔥 **Respuestas en streaming** (se escriben en tiempo real)
- 🎛️ **Barra lateral de configuración**: modelo, temperatura y personalidad del asistente
- 🧠 **Historial real de la conversación**: el modelo recibe los últimos 10 mensajes
- 🛡️ **Límite de mensajes por usuario y por día** + tope global diario (protege tu saldo de OpenAI)
- ✂️ Largo máximo de preguntas y respuestas para controlar costos
- 🔄 Botón de **nueva conversación**
- ⚠️ Manejo de errores: si OpenAI falla, el mensaje no se descuenta del límite

---

## 💻 Ejecutarlo en tu computador

Necesitas Python 3.10 o superior y una API Key de OpenAI.

```bash
pip install -r requirements.txt
```

Crea el archivo `.streamlit/secrets.toml` copiando `.streamlit/secrets.toml.example` y pon tu clave:

```toml
OPENAI_API_KEY = "sk-..."
```

Luego ejecuta:

```bash
streamlit run streamlit_chatbot.py
```

Se abre en `http://localhost:8501`.

> ⚠️ El archivo `secrets.toml` está en `.gitignore`: **nunca** subas tu API Key a GitHub.

---

## ☁️ Desplegarlo gratis en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
2. **Create app** → elige este repositorio, la rama `main` y el archivo `streamlit_chatbot.py`.
3. En **Advanced settings → Secrets** pega el contenido de `.streamlit/secrets.toml.example` con tu clave real.
4. **Deploy**. Obtendrás un enlace del tipo `https://tu-chatbot.streamlit.app`.

---

## ⚙️ Configuración (en Secrets)

| Variable | Qué hace | Por defecto |
|---|---|---|
| `OPENAI_API_KEY` | Tu clave de OpenAI | — |
| `MAX_MENSAJES_POR_USUARIO` | Mensajes que cada usuario puede enviar por día | `20` |
| `MAX_MENSAJES_GLOBALES_DIA` | Tope diario sumando a todos los usuarios | `300` |
| `MAX_TOKENS_RESPUESTA` | Largo máximo de cada respuesta | `1000` |

**Cómo funciona el límite:** cada usuario se identifica por su dirección IP (o por su sesión si no hay IP disponible). Los contadores se reinician cada día a la medianoche (hora de Colombia) y también cuando la app se reinicia. El tope global es la red de seguridad: aunque alguien cambie de red para saltarse el límite individual, el gasto diario total nunca pasa de ese número de mensajes.

---

## 🏗️ Tecnologías

- Python 3.10+
- Streamlit
- LangChain (LCEL, `ChatPromptTemplate`, `MessagesPlaceholder`)
- OpenAI API

---

## 🧪 En desarrollo

La carpeta `app/` contiene una versión "Pro" en construcción con memoria a largo plazo y búsqueda en PDFs (RAG). Todavía no está terminada y no se usa en el despliegue.
