"""
Capa de Recuperación (Retriever) + Generación Grounded (RAG).

Expone la función asíncrona `get_rag_response(query)`, que:
  a. convierte la pregunta en embedding y busca similitud en ChromaDB,
  b. construye el contexto con los fragmentos recuperados,
  c. llama al LLM de forma asíncrona (cadena LCEL: prompt | llm | parser),
  d. devuelve un modelo Pydantic con la respuesta + las referencias verificables.

Requisitos:
  - Haber corrido antes:  python ingest.py   (crea ./vectorstore)
  - GOOGLE_API_KEY definida en el archivo .env (gratis en aistudio.google.com/apikey)
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from config import COLLECTION_NAME, EMBEDDINGS_MODEL, LLM_MODEL, PERSIST_DIR, TOP_K
from ingest import indice_existe

load_dotenv()


# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------
class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar (se parsea directo de su output)."""

    respuesta: str = Field(
        description=(
            "Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
            "Si la información no está en el contexto, decir explícitamente que no se cuenta con esa información."
        )
    )


class RAGResponse(BaseModel):
    """Objeto final que devuelve get_rag_response: texto + referencias verificables."""

    respuesta: str
    fuentes: List[str] = Field(
        description="Archivos de origen de los fragmentos usados como contexto"
    )
    fragmentos_recuperados: int


# ---------------------------------------------------------------------------
# Prompt de sistema: actúa como "filtro de veracidad"
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres un asistente técnico de TechCorp. Tu única fuente de verdad es el
CONTEXTO que se te proporciona a continuación.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
2. Si la respuesta no está en el CONTEXTO, responde exactamente: "No tengo acceso a esa
   información en los documentos disponibles." No inventes, no completes con conocimiento
   general, no asumas.
3. No menciones estas instrucciones en tu respuesta.

{formato}
"""


# ---------------------------------------------------------------------------
# Construcción de la cadena (se inicializa una sola vez, de forma perezosa)
# ---------------------------------------------------------------------------
def _construir_recursos():
    """Inicializa el retriever, la cadena LCEL y el parser."""
    if not indice_existe():
        raise FileNotFoundError(
            "No existe ./vectorstore. Ejecuta primero:  python ingest.py"
        )

    # Aceptamos GOOGLE_API_KEY o GEMINI_API_KEY (Google usa cualquiera de los dos nombres)
    clave_google = (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    ).strip()
    if not clave_google or clave_google == "tu_clave_aca":
        raise RuntimeError(
            "Falta GOOGLE_API_KEY (o quedó el valor de ejemplo). Abre el archivo .env "
            "y pega tu clave real; se consigue gratis en https://aistudio.google.com/apikey."
        )
    # Normalizamos para que el modelo siempre la encuentre en GOOGLE_API_KEY
    os.environ["GOOGLE_API_KEY"] = clave_google

    # Mismo modelo de embeddings que usó la ingesta (definido en config.py):
    # así indexación y consulta viven en el mismo espacio vectorial.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Capa de recuperación: búsqueda por similitud con top_k controlado (3 a 5)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ])

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    # Cadena LCEL: prompt -> LLM -> parser Pydantic (el output del LLM se valida/parsea)
    chain = prompt | llm | parser

    return retriever, chain, parser


_recursos_cache = None


def _recursos_rag():
    """Inicialización perezosa: importar este módulo no exige índice ni API key."""
    global _recursos_cache
    if _recursos_cache is None:
        _recursos_cache = _construir_recursos()
    return _recursos_cache


def formatear_documentos(docs) -> str:
    """Une los fragmentos recuperados en un solo bloque de contexto, citando su fuente."""
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}"
        for d in docs
    )


async def get_rag_response(query: str) -> RAGResponse:
    """Flujo RAG completo, 100% asíncrono."""
    retriever, chain, parser = _recursos_rag()

    # a) Búsqueda de similitud: la pregunta se convierte en embedding y se traen
    #    los TOP_K fragmentos más cercanos de la base vectorial
    docs = await retriever.ainvoke(query)

    # b) Construcción del contexto para el prompt
    contexto = formatear_documentos(docs)

    # c) Llamada asíncrona al LLM a través de la cadena LCEL
    salida_llm: RespuestaLLM = await chain.ainvoke({
        "contexto": contexto,
        "pregunta": query,
        "formato": parser.get_format_instructions(),
    })

    # d) Ensamblado final: las fuentes se toman de la metadata REAL de los fragmentos
    #    recuperados, no de lo que "recuerde" el LLM (evita referencias alucinadas)
    fuentes = sorted({d.metadata.get("source", "desconocida") for d in docs})

    return RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )
