"""
Módulo de ingesta: carga los documentos de /data, aplica chunking estratégico
y los persiste en ChromaDB (./vectorstore).

Uso:
    python ingest.py

- La primera ejecución indexa los documentos.
- Las siguientes detectan el índice existente y NO reindexan (ahorra tiempo y cómputo).
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDINGS_MODEL,
    PERSIST_DIR,
)


def cargar_documentos() -> list:
    """Carga todos los archivos .txt de la carpeta /data."""
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos = loader.load()
    print(f"📄 Documentos cargados: {len(documentos)}")
    return documentos


def fragmentar(documentos: list) -> list:
    """
    Chunking en tokens (no en caracteres): chunk_size y chunk_overlap son tokens
    reales del tokenizer, así el límite de contexto del LLM se respeta de verdad.
    El solapamiento evita perder información en los bordes entre fragmentos.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos)
    print(f"✂️ Fragmentos generados: {len(chunks)}")
    return chunks


def indice_existe() -> bool:
    """True si ya hay un índice persistido con contenido (chequeo anti-reindexado)."""
    return os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0


def crear_vectorstore(chunks: list) -> Chroma:
    """Indexa los fragmentos en ChromaDB y los persiste en ./vectorstore."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )


def cargar_vectorstore() -> Chroma:
    """Carga el índice ya persistido, SIN reindexar, con el MISMO modelo de embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def main() -> None:
    if indice_existe():
        print("♻️ Índice existente detectado — se reutiliza sin reindexar")
        vectorstore = cargar_vectorstore()
    else:
        print("🆕 No hay índice previo — indexando documentos por primera vez")
        documentos = cargar_documentos()
        chunks = fragmentar(documentos)
        vectorstore = crear_vectorstore(chunks)

    print(f"📦 Documentos en la colección: {vectorstore._collection.count()}")


if __name__ == "__main__":
    main()
