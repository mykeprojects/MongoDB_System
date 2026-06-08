import re
from typing import Any

import numpy as np

from backend.models.rag_models import RetrievalHit
from backend.services.database_service import DatabaseService
from backend.services.embedding_service import EmbeddingService


class RetrievalService:
    EMBEDDING_COLLECTIONS = {
        "embedding_normatividad": ("normatividadId", "normatividad"),
        "embedding_productos": ("productoId", "producto"),
    }

    def __init__(self, database: DatabaseService, embeddings: EmbeddingService):
        self.database = database
        self.embeddings = embeddings

    def search(
        self,
        query: str,
        limit: int = 4,
        strategy: str | None = "semantico",
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []

        query_vector = self.embeddings.encode_one(query)
        hits: list[RetrievalHit] = []

        for collection, (resource_field, resource_type) in self.EMBEDDING_COLLECTIONS.items():
            filter_query: dict[str, Any] = {"activo": True}
            if strategy in {"frases", "semantico"}:
                filter_query["estrategiaChunking"] = strategy

            cursor = self.database.db[collection].find(
                filter_query,
                {
                    "titulo": 1,
                    "texto": 1,
                    "embedding": 1,
                    "estrategiaChunking": 1,
                    "chunkIndex": 1,
                    resource_field: 1,
                },
            )

            for document in cursor:
                embedding = np.asarray(document.get("embedding", []), dtype=float)
                if embedding.size == 0:
                    continue
                score = self.embeddings.cosine_similarity(query_vector, embedding)
                resource_id = str(document.get(resource_field, ""))
                hits.append(
                    RetrievalHit(
                        collection=collection,
                        title=str(document.get("titulo", "")),
                        text=str(document.get("texto", "")),
                        score=score,
                        chunk_index=int(document.get("chunkIndex", 0)),
                        strategy=str(document.get("estrategiaChunking", "")),
                        resource_id=resource_id,
                        resource_type=resource_type,
                        metadata={resource_field: resource_id},
                    )
                )

        hits = sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
        if hits:
            return hits

        return self._fallback_keyword_search(query, limit)

    def get_product_image(self, product_id: str | None) -> str | None:
        if not product_id:
            return None

        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            object_id = ObjectId(product_id)
        except InvalidId:
            return None

        product = self.database.db.productos.find_one(
            {"_id": object_id},
            {"imagenesUrl": 1, "nombre": 1},
        )
        if not product:
            return None

        images = product.get("imagenesUrl") or []
        if images:
            return str(images[0])
        return None

    def _fallback_keyword_search(self, query: str, limit: int) -> list[RetrievalHit]:
        tokens = [token for token in query.lower().split() if len(token) > 2]
        regex = "|".join(re.escape(token) for token in tokens[:6]) if tokens else re.escape(query)
        fallback_hits: list[RetrievalHit] = []

        product_filter: dict[str, Any] = {"activo": True}
        norm_filter: dict[str, Any] = {"activo": True}
        if regex:
            product_filter["$or"] = [
                {"nombre": {"$regex": regex, "$options": "i"}},
                {"descripcion": {"$regex": regex, "$options": "i"}},
            ]
            norm_filter["$or"] = [
                {"titulo": {"$regex": regex, "$options": "i"}},
                {"contenido": {"$regex": regex, "$options": "i"}},
                {"tipo": {"$regex": regex, "$options": "i"}},
            ]

        for document in self.database.db.productos.find(
            product_filter,
            {"nombre": 1, "descripcion": 1, "precio": 1, "stock": 1, "imagenesUrl": 1},
        ).limit(limit):
            text = (
                f"Producto: {document.get('nombre', '')}. "
                f"Descripcion: {document.get('descripcion', '')}. "
                f"Precio: {document.get('precio', '')}. Stock: {document.get('stock', '')}."
            )
            fallback_hits.append(
                RetrievalHit(
                    collection="productos",
                    title=str(document.get("nombre", "")),
                    text=text,
                    score=0.01,
                    chunk_index=0,
                    strategy="keyword-fallback",
                    resource_id=str(document.get("_id", "")),
                    resource_type="producto",
                    metadata={"imagenesUrl": document.get("imagenesUrl", [])},
                )
            )

        remaining = max(limit - len(fallback_hits), 0)
        if remaining:
            for document in self.database.db.normatividades.find(
                norm_filter,
                {"titulo": 1, "contenido": 1, "tipo": 1, "version": 1},
            ).limit(remaining):
                text = (
                    f"Normatividad: {document.get('titulo', '')}. "
                    f"Tipo: {document.get('tipo', '')}. Version: {document.get('version', '')}. "
                    f"Contenido: {document.get('contenido', '')}"
                )
                fallback_hits.append(
                    RetrievalHit(
                        collection="normatividades",
                        title=str(document.get("titulo", "")),
                        text=text,
                        score=0.01,
                        chunk_index=0,
                        strategy="keyword-fallback",
                        resource_id=str(document.get("_id", "")),
                        resource_type="normatividad",
                        metadata={},
                    )
                )

        return fallback_hits[:limit]
