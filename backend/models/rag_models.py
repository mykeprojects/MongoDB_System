from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalHit:
    collection: str
    title: str
    text: str
    score: float
    chunk_index: int
    strategy: str
    resource_id: str
    resource_type: str
    metadata: dict[str, Any]

    def to_source(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 4),
            "chunkIndex": self.chunk_index,
            "strategy": self.strategy,
            "resourceId": self.resource_id,
            "resourceType": self.resource_type,
        }


@dataclass
class ImageMatch:
    path: str
    title: str
    score: float


@dataclass
class MultimodalHit:
    imagen_id: str
    nombre_archivo: str
    ruta_imagen: str
    descripcion: str
    categoria: str
    etiquetas: list[str]
    score: float
