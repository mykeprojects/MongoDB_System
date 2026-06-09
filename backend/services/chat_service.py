from backend.models.chat_dto import ChatRequestDTO, ChatResponseDTO
from backend.models.rag_models import MultimodalHit, RetrievalHit
from backend.services.config_service import AppConfig
from backend.services.image_service import ImageService
from backend.services.llm_service import LLMService
from backend.services.multimodal_retrieval_service import MultimodalRetrievalService
from backend.services.retrieval_service import RetrievalService


class ChatService:
    def __init__(
        self,
        config: AppConfig,
        retrieval: RetrievalService,
        multimodal: MultimodalRetrievalService,
        llm: LLMService,
        images: ImageService,
    ):
        self.config = config
        self.retrieval = retrieval
        self.multimodal = multimodal
        self.llm = llm
        self.images = images

    def handle_chat(self, request: ChatRequestDTO) -> ChatResponseDTO:
        request.validate()

        mode = self._detect_mode(request)
        image_description = self.images.describe_image_path(request.image_path)
        query = " ".join(part for part in [request.message, image_description] if part).strip()

        multimodal_hits, hits = self._retrieve(request, mode, query)
        response_image = self._select_response_image(request, mode, multimodal_hits, query)

        question = request.message or "Describe la imagen recibida."
        response_text = self.llm.generate(
            question=question,
            hits=hits,
            image_context=image_description,
        )

        if mode == "image-to-text" and multimodal_hits:
            response_text = self._format_image_to_text_response(multimodal_hits, response_text)
        elif mode in {"text-to-image", "image-to-image"} and response_image and multimodal_hits:
            response_text = self._append_image_note(
                response_text,
                response_image,
                multimodal_hits[0],
            )

        return ChatResponseDTO(
            response=response_text,
            image_path=response_image,
            sources=[hit.to_source() for hit in hits],
            mode=mode,
        )

    def _retrieve(
        self,
        request: ChatRequestDTO,
        mode: str,
        query: str,
    ) -> tuple[list[MultimodalHit], list[RetrievalHit]]:
        limit = self.config.retrieval_limit

        if mode == "text-to-text":
            multimodal_hits = self.multimodal.search_text_to_text(query, limit=limit)
            if multimodal_hits:
                return multimodal_hits, MultimodalRetrievalService.to_retrieval_hits(multimodal_hits)
            rag_hits = self.retrieval.search(
                query=query,
                limit=limit,
                strategy=self.config.retrieval_strategy,
            )
            return [], rag_hits

        if mode == "text-to-image":
            multimodal_hits = self.multimodal.search_text_to_image(query or request.message, limit=limit)
            return multimodal_hits, MultimodalRetrievalService.to_retrieval_hits(multimodal_hits)

        if mode == "image-to-text":
            multimodal_hits = self.multimodal.search_image_to_text(request.image_path or "", limit=limit)
            return multimodal_hits, MultimodalRetrievalService.to_retrieval_hits(multimodal_hits)

        multimodal_hits = self.multimodal.search_image_to_image(request.image_path or "", limit=limit)
        return multimodal_hits, MultimodalRetrievalService.to_retrieval_hits(multimodal_hits)

    def _detect_mode(self, request: ChatRequestDTO) -> str:
        if request.image_path and request.want_image_response:
            return "image-to-image"
        if request.image_path:
            return "image-to-text"
        if request.want_image_response:
            return "text-to-image"
        return "text-to-text"

    def _select_response_image(
        self,
        request: ChatRequestDTO,
        mode: str,
        multimodal_hits: list[MultimodalHit],
        query: str,
    ) -> str | None:
        if mode not in {"text-to-image", "image-to-image"}:
            return None

        if multimodal_hits:
            return self.images.resolve_public_path(multimodal_hits[0].ruta_imagen)

        match = self.images.find_best_local_image(query, fallback=request.image_path)
        return match.path if match else None

    @staticmethod
    def _format_image_to_text_response(
        multimodal_hits: list[MultimodalHit],
        llm_response: str,
    ) -> str:
        best = multimodal_hits[0]
        header = (
            f"Descripción más similar ({best.categoria}, score={best.score:.4f}):\n"
            f"{best.descripcion}"
        )
        if llm_response.strip():
            return f"{header}\n\n{llm_response}"
        return header

    @staticmethod
    def _append_image_note(
        response_text: str,
        image_path: str,
        best_hit: MultimodalHit,
    ) -> str:
        note = (
            f"\n\nImagen relacionada: {image_path}\n"
            f"Coincidencia: {best_hit.descripcion}"
        )
        return f"{response_text}{note}"
