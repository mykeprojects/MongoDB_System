import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymongo
import pypdf
import tkinter as tk
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from sentence_transformers import SentenceTransformer
from tkinter import filedialog

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ingest_documents")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
MODEL_NAME = "all-MiniLM-L6-v2"

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("Las variables MONGO_URI y DATABASE_NAME deben estar definidas en el archivo .env")

CHUNK_SIZE_SENTENCES = 5
CHUNK_OVERLAP_SENTENCES = 1
SEMANTIC_SIMILARITY_THRESHOLD = 0.5


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def common_fields() -> dict[str, Any]:
    date = now_utc()
    return {
        "schemaVersion": 1,
        "creadoEn": date,
        "actualizadoEn": date,
        "eliminadoEn": None,
    }


def lector_archivos() -> str | None:
    root = tk.Tk()
    root.withdraw()

    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona un Archivo de Texto",
        filetypes=[("Todos los archivos", "*.*")],
    )

    if not ruta_archivo:
        logger.warning("No se seleccionó ningún archivo.")
        return None

    ruta = Path(ruta_archivo)
    extension = ruta.suffix.lower()

    if extension == ".txt":
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            return archivo.read()

    if extension == ".pdf":
        lector = pypdf.PdfReader(ruta_archivo)
        contenido = ""
        for pagina in lector.pages:
            contenido += pagina.extract_text(extraction_mode="layout") + "\n"
        return contenido

    logger.error("Formato no soportado: %s", extension)
    return None


def solicitar_datos() -> dict[str, Any]:
    titulo = input("Ingrese el título del archivo: ").strip()
    while not titulo:
        titulo = input("El título no puede estar vacío. Ingrese el título del archivo: ").strip()

    while True:
        print("Categorías disponibles: normatividad, productos")
        categoria = input("Ingrese la categoría del documento: ").strip().lower()
        if categoria in ("normatividad", "productos"):
            break
        print("Categoría inválida. Use 'normatividad' o 'productos'.")

    if categoria == "normatividad":
        recurso_id = input("Ingrese el ID de la normatividad: ").strip()
        coleccion = "embedding_normatividad"
        campo_id = "normatividadId"
    else:
        recurso_id = input("Ingrese el ID del producto: ").strip()
        coleccion = "embedding_productos"
        campo_id = "productoId"

    try:
        object_id = ObjectId(recurso_id)
    except InvalidId as exc:
        raise ValueError(f"El ID proporcionado no es un ObjectId válido: {recurso_id}") from exc

    return {
        "titulo": titulo,
        "categoria": categoria,
        "coleccion": coleccion,
        "campo_id": campo_id,
        "recurso_id": object_id,
    }


def dividir_en_frases(texto: str) -> list[str]:
    texto = re.sub(r"\s+", " ", texto.strip())
    if not texto:
        return []

    frases = re.split(r"(?<=[.!?])\s+", texto)
    return [frase.strip() for frase in frases if frase.strip()]


def chunking_por_frases(
    texto: str,
    tamano: int = CHUNK_SIZE_SENTENCES,
    overlap: int = CHUNK_OVERLAP_SENTENCES,
) -> list[str]:
    frases = dividir_en_frases(texto)
    if not frases:
        return []

    if len(frases) <= tamano:
        return [" ".join(frases)]

    paso = tamano - overlap
    chunks: list[str] = []

    for inicio in range(0, len(frases), paso):
        bloque = frases[inicio : inicio + tamano]
        if bloque:
            chunks.append(" ".join(bloque))
        if inicio + tamano >= len(frases):
            break

    return chunks


def _similitud_coseno(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def chunking_semantico(
    texto: str,
    model: SentenceTransformer,
    umbral: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[str]:
    frases = dividir_en_frases(texto)
    if not frases:
        return []
    if len(frases) == 1:
        return [frases[0]]

    embeddings = model.encode(frases, convert_to_numpy=True)
    chunks: list[str] = []
    bloque_actual = [frases[0]]

    for indice in range(1, len(frases)):
        similitud = _similitud_coseno(embeddings[indice - 1], embeddings[indice])
        if similitud < umbral:
            chunks.append(" ".join(bloque_actual))
            bloque_actual = [frases[indice]]
        else:
            bloque_actual.append(frases[indice])

    if bloque_actual:
        chunks.append(" ".join(bloque_actual))

    return chunks


def generar_embeddings(textos: list[str], model: SentenceTransformer) -> list[list[float]]:
    if not textos:
        return []
    vectores = model.encode(textos, convert_to_numpy=True)
    return [vector.astype(float).tolist() for vector in vectores]


def construir_documentos(
    chunks: list[str],
    embeddings: list[list[float]],
    estrategia: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    documentos: list[dict[str, Any]] = []

    for indice, (texto, embedding) in enumerate(zip(chunks, embeddings)):
        documento = {
            metadata["campo_id"]: metadata["recurso_id"],
            "titulo": metadata["titulo"],
            "texto": texto,
            "embedding": embedding,
            "estrategiaChunking": estrategia,
            "chunkIndex": indice,
            "activo": True,
            **common_fields(),
        }
        documentos.append(documento)

    return documentos


def guardar_chunks(db, coleccion: str, documentos: list[dict[str, Any]]) -> int:
    if not documentos:
        return 0

    resultado = db[coleccion].insert_many(documentos, ordered=False)
    return len(resultado.inserted_ids)


def procesar_documento() -> None:
    metadata = solicitar_datos()
    contenido = lector_archivos()

    if not contenido or not contenido.strip():
        logger.error("No se obtuvo contenido válido del archivo seleccionado.")
        return

    logger.info("Cargando modelo SentenceTransformer: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    chunks_frases = chunking_por_frases(contenido)
    chunks_semanticos = chunking_semantico(contenido, model)

    logger.info("Chunks por frases: %s", len(chunks_frases))
    logger.info("Chunks semánticos: %s", len(chunks_semanticos))

    embeddings_frases = generar_embeddings(chunks_frases, model)
    embeddings_semanticos = generar_embeddings(chunks_semanticos, model)

    documentos = []
    documentos.extend(
        construir_documentos(chunks_frases, embeddings_frases, "frases", metadata)
    )
    documentos.extend(
        construir_documentos(chunks_semanticos, embeddings_semanticos, "semantico", metadata)
    )

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[DATABASE_NAME]
        insertados = guardar_chunks(db, metadata["coleccion"], documentos)
        logger.info(
            "Ingesta completada. %s chunks guardados en '%s' para %s=%s",
            insertados,
            metadata["coleccion"],
            metadata["campo_id"],
            metadata["recurso_id"],
        )
    except ConnectionFailure as exc:
        logger.error("No se pudo conectar a MongoDB: %s", exc)
        raise
    except pymongo.errors.BulkWriteError as exc:
        # Extrae cuántos documentos sí se lograron guardar con éxito
        insertados = exc.details.get("nInserted", 0)
        logger.warning(
            "Se guardaron %s chunks, pero ocurrieron errores de validación en otros.", 
            insertados
        )
    except PyMongoError as exc:
        logger.error("Error crítico de MongoDB: %s", exc)
        raise


if __name__ == "__main__":
    procesar_documento()
