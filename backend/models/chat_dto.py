from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChatRequestDTO:
    message: str = ""
    image_path: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ChatRequestDTO":
        payload = payload or {}
        message = payload.get("message") or ""
        image_path = payload.get("imagePath") or payload.get("image_path")

        return cls(
            message=str(message).strip(),
            image_path=str(image_path).strip() if image_path else None,
        )

    def validate(self) -> None:
        if not self.message and not self.image_path:
            raise ValueError("Debes enviar un mensaje, una imagen, o ambos.")


@dataclass
class ChatResponseDTO:
    response: str
    image_path: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "text-to-text"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["imagePath"] = payload.pop("image_path")
        return payload
