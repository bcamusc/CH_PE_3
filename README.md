# 🧠 Pre-entrega 3 — Sistema de recuperación semántica local (RAG)

Sistema **RAG end-to-end** que responde preguntas sobre las políticas internas de *TechCorp* usando **exclusivamente** los documentos de `data/`. Si la respuesta no está en los documentos, el sistema lo dice explícitamente en lugar de inventarla (anti-alucinación).

## Stack

| Rol | Tecnología |
|-----|------------|
| Orquestación | LangChain — cadena **LCEL asíncrona** |
| Base vectorial | **ChromaDB** persistente en `./vectorstore` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, sin API) |
| LLM | Google **Gemini** (`gemini-flash-latest`, capa gratuita) |
| Salida estructurada | **Pydantic** vía `PydanticOutputParser` |

## Estructura del repositorio

```
.
├── data/                                # Dataset de ejemplo (4 políticas de TechCorp en .txt)
├── config.py                            # Configuración central (una sola fuente de verdad)
├── ingest.py                            # 1) Ingesta: carga, chunking y persistencia en ChromaDB
├── rag.py                               # 2) Cadena RAG asíncrona (get_rag_response) + esquemas Pydantic
├── main.py                              # 3) Demo: pruebas automáticas + modo interactivo
├── requirements.txt                     # Dependencias
├── .env.example                         # Plantilla de variables de entorno (sin claves reales)
└── .gitignore                           # Excluye .env, .venv, vectorstore/ y el notebook de clases
```

## Cómo ejecutar

### 1) Crear entorno virtual e instalar dependencias

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 💡 Si PowerShell bloquea la activación, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` o usa CMD con `.\.venv\Scripts\activate.bat`.
> La primera ejecución descarga el modelo de embeddings (~90 MB) desde HuggingFace.

### 2) Configurar la API key (nunca en el código)

```powershell
Copy-Item .env.example .env    # macOS/Linux: cp .env.example .env
```

Edita `.env` y pega tu clave gratuita de [Google AI Studio](https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY=tu_clave_real
```

`.env` está incluido en `.gitignore`, así que **nunca se sube al repositorio**.

### 3) Indexar los documentos (solo la primera vez)

```bash
python ingest.py
```

- Primera ejecución: fragmenta los `.txt` de `data/` (chunks de 500 tokens con 50 de overlap) y persiste el índice en `./vectorstore`.
- Ejecuciones siguientes: detecta que el índice ya existe y **no reindexa**.

### 4) Ejecutar las pruebas y el modo interactivo

```bash
python main.py          # corre las 2 pruebas y luego abre el chat por consola
python main.py --test   # corre solo las 2 pruebas automáticas
```

## Pruebas incluidas

| # | Tipo | Pregunta | Resultado esperado |
|---|------|----------|--------------------|
| 1 | Respuesta en contexto | ¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad? | Responde **21 días corridos** y cita `data/politica_vacaciones.txt` |
| 2 | Pregunta trampa | ¿Cuál es la política de bonos por rendimiento anual en TechCorp? | Responde **"No tengo acceso a esa información en los documentos disponibles."** — sin inventar |

## Decisiones de diseño (errores comunes evitados)

- **Embeddings coincidentes:** el mismo modelo se usa para indexar y consultar, definido una sola vez en `config.py`.
- **Top-k controlado:** `TOP_K = 4` (entre 3 y 5). Evita el *"contexto infinito"* y la degradación por *Lost in the Middle*.
- **Persistencia con chequeo:** `ingest.py` verifica si `./vectorstore` ya existe antes de reindexar.
- **Chunking en tokens:** `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)`. El límite del LLM se mide en tokens, no en caracteres. *(chunk_size es un techo: con documentos cortos puede resultar 1 fragmento por archivo, y está bien).*
- **Referencias verificables:** las fuentes del resultado se arman desde la metadata real de los fragmentos recuperados; no se le pide al LLM que "recuerde" de dónde sacó la información.
- **Filtro de veracidad:** el prompt de sistema obliga a responder solo con el CONTEXTO y a usar una frase exacta cuando la información no está.
- **Todo asíncrono:** `get_rag_response()` usa `await` tanto en el retriever (`ainvoke`) como en la cadena LCEL.

## Publicar en GitHub

```bash
git init -b main
git add .
git commit -m "Pre-entrega 3: sistema RAG local"
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

Antes de subir, verifica que `.env` **no** aparezca en `git status` (debe estar ignorado por `.gitignore`).

## Notas

- El notebook de clases queda fuera de este repositorio a propósito; todo el flujo del sistema está en `ingest.py`, `rag.py` y `main.py`.
- Para usar otro proveedor de LLM (por ejemplo OpenAI), reemplaza `ChatGoogleGenerativeAI` por `ChatOpenAI` en `rag.py` y agrega la variable de entorno correspondiente.
- (Opcional) Si el proyecto vive dentro de OneDrive, considera crear `.venv` fuera de OneDrive (por ejemplo `C:\venvs\pe3`) para evitar que se sincronicen miles de archivos de librerías.
