from dataclasses import asdict, dataclass, field
from typing import Any

#PETICION de USUARIO
@dataclass
class ChatRequestDTO:
    message: str = ""
    image_path: str | None = None
    want_image_response: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ChatRequestDTO":
        payload = payload or {}
        message = payload.get("message") or ""
        image_path = payload.get("imagePath") or payload.get("image_path")
        want_image_response = payload.get("wantImageResponse")
        if want_image_response is None:
            want_image_response = payload.get("want_image_response")

        return cls(
            #extrae el mensaje y ruta de imagen (si hay), y actualiza la clase en sus atributos
            message=str(message).strip(),
            image_path=str(image_path).strip() if image_path else None,
            want_image_response=bool(want_image_response),
        )

    def validate(self) -> None:
        if not self.message and not self.image_path:
            raise ValueError("Debes enviar un mensaje, una imagen, o ambos.")

#RESPUESTA del RAG
@dataclass
class ChatResponseDTO:
    response: str
    image_path: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "text-to-text"
    #por defecto la respuesta es texto a texto.

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["imagePath"] = payload.pop("image_path")
        return payload
