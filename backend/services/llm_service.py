import json
import urllib.error
import urllib.request

from backend.models.rag_models import RetrievalHit
from backend.services.config_service import AppConfig


class LLMService:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(self, question: str, hits: list[RetrievalHit], image_context: str = "") -> str:
        context = self._build_context(hits, image_context)
        if self.config.groq_api_key:
            try:
                return self._generate_with_groq(question, context)
            except Exception:
                return self._fallback_answer(question, hits, image_context)
        return self._fallback_answer(question, hits, image_context)

    def _generate_with_groq(self, question: str, context: str) -> str:
        body = {
            "model": self.config.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente RAG para un e-commerce NoSQL. "
                        "Responde en espanol, usando solo el contexto recuperado. "
                        "Si falta informacion, dilo de forma breve."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta:\n{question}",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }

        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Groq error {exc.code}: {detail}") from exc

        return payload["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _build_context(hits: list[RetrievalHit], image_context: str) -> str:
        sections = []
        if image_context:
            sections.append(f"Imagen recibida: {image_context}")
        for index, hit in enumerate(hits, start=1):
            sections.append(
                f"[{index}] {hit.resource_type} | {hit.title} | score={hit.score:.4f}\n{hit.text}"
            )
        return "\n\n".join(sections) or "No hay contexto recuperado."

    @staticmethod
    def _fallback_answer(question: str, hits: list[RetrievalHit], image_context: str) -> str:
        if not hits:
            if image_context:
                return (
                    f"Recibi la imagen '{image_context}', pero no encontre contexto suficiente "
                    "en MongoDB para responder con precision."
                )
            return "No encontre contexto suficiente en MongoDB para responder esa consulta."

        if hits[0].strategy == "keyword-fallback":
            lines = ["Encontre estos datos en MongoDB:"]
            for hit in hits:
                lines.append(f"- {hit.title}: {hit.text}")
            if image_context:
                lines.insert(0, f"Use tambien la referencia de imagen: {image_context}.")
            return "\n".join(lines)

        best = hits[0]
        prefix = ""
        if image_context:
            prefix = f"Con la imagen recibida ({image_context}) y el contexto recuperado: "

        return (
            f"{prefix}{best.text}\n\n"
            f"Fuente principal: {best.title} ({best.resource_type}, similitud {best.score:.2f})."
        )
