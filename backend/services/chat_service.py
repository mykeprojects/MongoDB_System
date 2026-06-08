from backend.models.chat_dto import ChatRequestDTO, ChatResponseDTO
from backend.models.rag_models import RetrievalHit
from backend.services.config_service import AppConfig
from backend.services.image_service import ImageService
from backend.services.llm_service import LLMService
from backend.services.retrieval_service import RetrievalService


class ChatService:
    IMAGE_INTENT_WORDS = {
        "imagen",
        "foto",
        "muestrame",
        "muéstrame",
        "ver",
        "visual",
        "similar",
    }

    def __init__(
        self,
        config: AppConfig,
        retrieval: RetrievalService,
        llm: LLMService,
        images: ImageService,
    ):
        self.config = config
        self.retrieval = retrieval
        self.llm = llm
        self.images = images

    def handle_chat(self, request: ChatRequestDTO) -> ChatResponseDTO:
        request.validate()

        image_description = self.images.describe_image_path(request.image_path)
        query = " ".join(part for part in [request.message, image_description] if part).strip()
        mode = self._detect_mode(request)
        hits = self.retrieval.search(
            query=query,
            limit=self.config.retrieval_limit,
            strategy=self.config.retrieval_strategy,
        )

        response_image = self._select_response_image(request, hits, query, mode)
        response_text = self.llm.generate(
            question=request.message or "Describe la imagen recibida.",
            hits=hits,
            image_context=image_description,
        )

        if mode in {"text-to-image", "image-to-image"} and response_image:
            response_text = self._append_image_note(response_text, response_image)

        return ChatResponseDTO(
            response=response_text,
            image_path=response_image,
            sources=[hit.to_source() for hit in hits],
            mode=mode,
        )

    def _detect_mode(self, request: ChatRequestDTO) -> str:
        wants_image = self._looks_like_image_request(request.message)
        if request.image_path and wants_image:
            return "image-to-image"
        if request.image_path:
            return "image-to-text"
        if wants_image:
            return "text-to-image"
        return "text-to-text"

    def _select_response_image(
        self,
        request: ChatRequestDTO,
        hits: list[RetrievalHit],
        query: str,
        mode: str,
    ) -> str | None:
        if mode not in {"text-to-image", "image-to-image"}:
            return None

        for hit in hits:
            if hit.resource_type == "producto":
                image_url = self.retrieval.get_product_image(hit.resource_id)
                if image_url:
                    return image_url

        match = self.images.find_best_local_image(query, fallback=request.image_path)
        return match.path if match else None

    def _looks_like_image_request(self, message: str) -> bool:
        text = (message or "").lower()
        return any(word in text for word in self.IMAGE_INTENT_WORDS)

    @staticmethod
    def _append_image_note(response_text: str, image_path: str) -> str:
        if image_path.startswith("http"):
            return f"{response_text}\n\nTambien encontre una imagen asociada al producto."
        return f"{response_text}\n\nTambien encontre una imagen local relacionada: {image_path}."
