"""
Configuración central del sistema RAG de TechCorp.

Una sola fuente de verdad para los parámetros clave. En particular, el modelo
de embeddings se define UNA vez porque debe ser el mismo al indexar (ingest.py)
y al consultar (rag.py): si cambia entre ambos, la distancia vectorial pierde
sentido y la búsqueda devuelve resultados aleatorios (el error #1 más común).
"""

# --- Rutas ---
DATA_DIR = "data"
PERSIST_DIR = "./vectorstore"

# --- Base vectorial ---
COLLECTION_NAME = "techcorp_policies"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking en tokens (no en caracteres) ---
# La consigna pide mínimo 500 tokens con 50 de solapamiento.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Recuperación ---
TOP_K = 4  # entre 3 y 5: suficiente contexto sin caer en el "contexto infinito"

# --- LLM ---
LLM_MODEL = "gemini-flash-latest"
