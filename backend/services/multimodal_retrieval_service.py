import logging
from typing import Any

import numpy as np
from pymongo.errors import OperationFailure

from backend.models.rag_models import MultimodalHit, RetrievalHit
from backend.services.clip_embedding_service import ClipEmbeddingService
from backend.services.config_service import AppConfig, IMAGES_DIR
from backend.services.database_service import DatabaseService
from backend.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class MultimodalRetrievalService:
    COLLECTION = "embedding_multimodal"

    def __init__(
        self,
        database: DatabaseService,
        embeddings: EmbeddingService,
        clip_embeddings: ClipEmbeddingService,
        config: AppConfig,
    ):
        self.database = database
        self.embeddings = embeddings
        self.clip_embeddings = clip_embeddings
        self.config = config

    def search_text_to_text(self, query: str, limit: int | None = None) -> list[MultimodalHit]:
        vector = self.embeddings.encode_one(query)
        return self._search(
            vector=vector,
            index_name=self.config.multimodal_text_index,
            path="embeddingTexto",
            limit=limit,
        )

    def search_text_to_image(self, query: str, limit: int | None = None) -> list[MultimodalHit]:
        vector = self.clip_embeddings.encode_text(query)
        return self._search(
            vector=vector,
            index_name=self.config.multimodal_clip_image_index,
            path="embeddingImagen",
            limit=limit,
        )

    def search_image_to_text(self, image_path: str, limit: int | None = None) -> list[MultimodalHit]:
        vector = self._encode_query_image(image_path)
        return self._search(
            vector=vector,
            index_name=self.config.multimodal_clip_text_index,
            path="embeddingTextoClip",
            limit=limit,
        )

    def search_image_to_image(self, image_path: str, limit: int | None = None) -> list[MultimodalHit]:
        vector = self._encode_query_image(image_path)
        return self._search(
            vector=vector,
            index_name=self.config.multimodal_clip_image_index,
            path="embeddingImagen",
            limit=limit,
        )

    def _encode_query_image(self, image_path: str) -> np.ndarray:
        from pathlib import Path

        normalized = image_path.replace("\\", "/").strip()
        filename = Path(normalized).name
        local_path = IMAGES_DIR / filename
        if not local_path.exists():
            raise ValueError(f"No se encontró la imagen local: {filename}")
        return self.clip_embeddings.encode_image_path(str(local_path))

    def _search(
        self,
        vector: np.ndarray,
        index_name: str,
        path: str,
        limit: int | None,
    ) -> list[MultimodalHit]:
        limit = limit or self.config.retrieval_limit
        filter_query: dict[str, Any] = {"activo": {"$eq": True}}

        hits = self._vector_search(vector, index_name, path, filter_query, limit)
        if hits:
            return hits

        logger.warning(
            "Vector search multimodal falló en %s; usando similitud coseno local.",
            path,
        )
        return self._local_search(vector, path, filter_query, limit)

    def _vector_search(
        self,
        vector: np.ndarray,
        index_name: str,
        path: str,
        filter_query: dict[str, Any],
        limit: int,
    ) -> list[MultimodalHit]:
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": path,
                    "queryVector": vector.tolist(),
                    "numCandidates": max(limit * 20, 100),
                    "limit": limit,
                    "filter": filter_query,
                }
            },
            {
                "$project": {
                    "imagenId": 1,
                    "nombreArchivo": 1,
                    "rutaImagen": 1,
                    "descripcion": 1,
                    "etiquetas": 1,
                    "categoria": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            documents = list(self.database.db[self.COLLECTION].aggregate(pipeline))
        except OperationFailure as exc:
            logger.warning("Atlas vector search multimodal no disponible: %s", exc)
            return []

        return [self._to_hit(document) for document in documents]

    def _local_search(
        self,
        vector: np.ndarray,
        path: str,
        filter_query: dict[str, Any],
        limit: int,
    ) -> list[MultimodalHit]:
        mongo_filter = {"activo": True}
        hits: list[MultimodalHit] = []

        for document in self.database.db[self.COLLECTION].find(
            mongo_filter,
            {
                "imagenId": 1,
                "nombreArchivo": 1,
                "rutaImagen": 1,
                "descripcion": 1,
                "etiquetas": 1,
                "categoria": 1,
                path: 1,
            },
        ):
            embedding = np.asarray(document.get(path, []), dtype=float)
            if embedding.size == 0:
                continue
            score = self.embeddings.cosine_similarity(vector, embedding)
            document["score"] = score
            hits.append(self._to_hit(document))

        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def _to_hit(self, document: dict[str, Any]) -> MultimodalHit:
        return MultimodalHit(
            imagen_id=str(document.get("imagenId", "")),
            nombre_archivo=str(document.get("nombreArchivo", "")),
            ruta_imagen=str(document.get("rutaImagen", "")),
            descripcion=str(document.get("descripcion", "")),
            categoria=str(document.get("categoria", "")),
            etiquetas=list(document.get("etiquetas", [])),
            score=float(document.get("score", 0.0)),
        )

    @staticmethod
    def to_retrieval_hits(multimodal_hits: list[MultimodalHit]) -> list[RetrievalHit]:
        retrieval_hits: list[RetrievalHit] = []
        for hit in multimodal_hits:
            retrieval_hits.append(
                RetrievalHit(
                    collection="embedding_multimodal",
                    title=hit.nombre_archivo,
                    text=hit.descripcion,
                    score=hit.score,
                    chunk_index=0,
                    strategy="multimodal",
                    resource_id=hit.imagen_id,
                    resource_type="imagen",
                    metadata={
                        "rutaImagen": hit.ruta_imagen,
                        "categoria": hit.categoria,
                        "etiquetas": hit.etiquetas,
                    },
                )
            )
        return retrieval_hits
