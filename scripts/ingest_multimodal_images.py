"""
Ingesta de imágenes locales con embeddings multimodales (MiniLM + CLIP).

- Archivos cuyo nombre comienza por 'adidas' -> descripciones de tenis.
- Resto de imágenes -> descripciones de tecnología (únicas entre sí).
"""

import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ingest_multimodal_images")

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
IMAGES_DIR = ROOT_DIR / "data" / "images"
COLLECTION_NAME = "embedding_multimodal"

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("Las variables MONGO_URI y DATABASE_NAME deben estar definidas en el archivo .env")

TENIS_TEMPLATES = [
    "Tenis {modelo} en color {color}, con suela de goma antideslizante y diseño urbano casual.",
    "Par de zapatillas {modelo} estilo retro, upper en material sintético y plantilla acolchada.",
    "Calzado deportivo {modelo} para uso diario, con costuras reforzadas y perfil bajo.",
    "Sneakers {modelo} con puntera redondeada, cordones planos y acabado mate en tono {color}.",
    "Tenis {modelo} inspirados en siluetas clásicas, ligeros y cómodos para caminar.",
    "Zapatillas {modelo} con entresuela de EVA y detalle lateral icónico de la marca.",
    "Modelo {modelo} de calzado casual, ideal para combinar con jeans y outfits streetwear.",
    "Tenis {modelo} con empeine flexible, suela cupsole y estética minimalista en {color}.",
]

TECNO_TEMPLATES = [
    "Dispositivo {tipo} {modelo} con pantalla de alta resolución y batería de larga duración.",
    "Equipo {tipo} {modelo} orientado a productividad, con procesador eficiente y almacenamiento rápido.",
    "Gadget {tipo} {modelo} con conectividad inalámbrica avanzada y diseño compacto.",
    "Producto tecnológico {tipo} {modelo}, pensado para multimedia, trabajo y entretenimiento.",
    "Dispositivo {tipo} {modelo} con cámara mejorada, sensores precisos y acabado premium.",
    "Equipo electrónico {tipo} {modelo} con interfaz fluida y compatibilidad con ecosistemas modernos.",
    "Aparato {tipo} {modelo} liviano, con panel nítido y rendimiento equilibrado para el día a día.",
    "Solución {tipo} {modelo} con hardware optimizado para tareas exigentes y uso prolongado.",
]

COLORS = ["blanco", "negro", "gris", "azul", "verde", "rojo", "beige", "crema"]


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


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "imagen"


def is_adidas(filename: str) -> bool:
    return filename.lower().startswith("adidas")


def extract_tenis_model(stem: str) -> str:
    cleaned = re.sub(r"^adidas[\s_\-()]*", "", stem, flags=re.IGNORECASE).strip(" _-()")
    if not cleaned:
        return "Originals"
    return cleaned.replace("_", " ").replace("-", " ").strip()


def extract_tech_info(stem: str) -> tuple[str, str]:
    normalized = stem.replace("_", " ")
    if normalized.lower().startswith("computador"):
        return "computador portátil", normalized.replace("Computador", "").strip() or "Asus"
    if normalized.lower().startswith("mackbook") or normalized.lower().startswith("macbook"):
        return "laptop", normalized
    if normalized.lower().startswith("iphone"):
        return "smartphone", normalized
    if normalized.lower().startswith("celular"):
        return "teléfono móvil", normalized
    if normalized.lower().startswith("ipad") or normalized.lower().startswith("iphad"):
        return "tablet", normalized
    return "dispositivo electrónico", normalized


def build_description(filename: str, index: int) -> tuple[str, str, list[str]]:
    stem = Path(filename).stem

    if is_adidas(filename):
        modelo = extract_tenis_model(stem)
        color = COLORS[index % len(COLORS)]
        template = TENIS_TEMPLATES[index % len(TENIS_TEMPLATES)]
        descripcion = template.format(modelo=modelo, color=color)
        etiquetas = ["tenis", "zapatillas", "adidas", modelo.lower(), color]
        return descripcion, "tenis", etiquetas

    tipo, modelo = extract_tech_info(stem)
    template = TECNO_TEMPLATES[index % len(TECNO_TEMPLATES)]
    descripcion = template.format(tipo=tipo, modelo=modelo)
    etiquetas = ["tecnologia", tipo.replace(" ", "_"), modelo.lower().replace(" ", "_")]
    return descripcion, "tecnologia", etiquetas


class MultimodalEmbedder:
    def __init__(self) -> None:
        logger.info("Cargando MiniLM (%s)...", MODEL_NAME)
        self.text_model = SentenceTransformer(MODEL_NAME)
        logger.info("Cargando CLIP (%s)...", CLIP_MODEL_NAME)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.clip_model.eval()

    def embed_text_minilm(self, text: str) -> list[float]:
        vector = self.text_model.encode(text)
        return np.asarray(vector, dtype=np.float32).tolist()

    def embed_text_clip(self, text: str) -> list[float]:
        inputs = self.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
            text_features = self.clip_model.text_projection(pooled)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().astype(np.float32).tolist()

    def embed_image_clip(self, image: Image.Image) -> list[float]:
        image = image.convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
            image_features = self.clip_model.visual_projection(pooled)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features[0].cpu().numpy().astype(np.float32).tolist()


def build_document(file_path: Path, index: int, embedder: MultimodalEmbedder) -> dict[str, Any]:
    descripcion, categoria, etiquetas = build_description(file_path.name, index)
    imagen_id = slugify(file_path.stem)
    ruta_imagen = f"data/images/{file_path.name}"

    with Image.open(file_path) as pil_image:
        embedding_imagen = embedder.embed_image_clip(pil_image)

    return {
        "imagenId": imagen_id,
        "nombreArchivo": file_path.name,
        "rutaImagen": ruta_imagen,
        "descripcion": descripcion,
        "etiquetas": etiquetas,
        "categoria": categoria,
        "embeddingImagen": embedding_imagen,
        "embeddingTextoClip": embedder.embed_text_clip(descripcion),
        "embeddingTexto": embedder.embed_text_minilm(descripcion),
        "activo": True,
        **common_fields(),
    }


def ingest_images() -> int:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes: {IMAGES_DIR}")

    image_files = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not image_files:
        logger.warning("No hay imágenes para procesar en %s", IMAGES_DIR)
        return 0

    embedder = MultimodalEmbedder()
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    inserted = 0
    for index, file_path in enumerate(image_files):
        document = build_document(file_path, index, embedder)
        result = collection.update_one(
            {"imagenId": document["imagenId"]},
            {"$set": document},
            upsert=True,
        )
        if result.upserted_id or result.modified_count:
            inserted += 1
        logger.info(
            "[%s] %s -> %s",
            document["categoria"],
            file_path.name,
            document["descripcion"][:80],
        )

    client.close()
    return inserted


def main() -> None:
    try:
        total = ingest_images()
        logger.info("Ingesta multimodal finalizada. Documentos procesados: %s", total)
    except ConnectionFailure as exc:
        logger.error("No se pudo conectar a MongoDB: %s", exc)
        raise
    except PyMongoError as exc:
        logger.error("Error de MongoDB durante la ingesta: %s", exc)
        raise


if __name__ == "__main__":
    main()
