import json
import logging
import urllib.error
import urllib.request

from backend.models.rag_models import RetrievalHit
from backend.services.config_service import AppConfig

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(self, question: str, hits: list[RetrievalHit], image_context: str = "") -> str:
        context = self._build_context(hits, image_context)
        if self.config.groq_api_key:
            try:
                answer = self._generate_with_groq(question, context)
                logger.info(
                    "Respuesta generada con Groq (%s) usando %s fragmentos de contexto.",
                    self.config.groq_model,
                    len(hits),
                )
                return answer
            except Exception as exc:
                logger.warning("Groq fallo, usando respuesta local: %s", exc)
                return self._fallback_answer(question, hits, image_context)
        logger.warning("GROQ_API_KEY no configurada; usando respuesta local sin LLM.")
        return self._fallback_answer(question, hits, image_context)

    def _generate_with_groq(self, question: str, context: str) -> str:
        body = {
            "model": self.config.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        """Eres un asistente especializado en responder consultas utilizando información recuperada mediante un sistema RAG.

                        Tu objetivo es responder la pregunta del usuario de forma clara, completa y bien redactada, utilizando exclusivamente la información presente en los fragmentos de contexto proporcionados.

                        Instrucciones:

                        1. Analiza primero la pregunta para identificar exactamente qué información solicita el usuario.
                        2. Utiliza los fragmentos recuperados para construir una respuesta coherente y natural, no una simple copia de texto.
                        3. Sintetiza, organiza y relaciona la información relevante proveniente de múltiples fragmentos cuando sea necesario.
                        4. Explica los conceptos de manera clara y contextualizada, desarrollando la respuesta con base en la información disponible.
                        5. Puedes reformular el contenido para mejorar la comprensión, pero nunca agregar información que no esté respaldada por el contexto.
                        6. Si la pregunta se refiere a una norma, procedimiento, política o regulación, explica su propósito, alcance, requisitos o características utilizando únicamente la información encontrada.
                        7. Si existen varios fragmentos relacionados, integra sus aportes en una única respuesta estructurada.
                        8. Mantén un tono profesional y descriptivo.
                        9. No inventes datos, fechas, requisitos, definiciones ni conclusiones.
                        10. Si la información disponible es insuficiente para responder completamente, indica qué aspectos sí están documentados y cuáles no aparecen en los fragmentos recuperados."""
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
            chunk_text = hit.text.strip()
            if not chunk_text:
                logger.warning(
                    "Fragmento %s (%s) sin campo 'texto'; solo se incluira el titulo.",
                    index,
                    hit.title,
                )
            sections.append(
                f"[{index}] {hit.resource_type} | score={hit.score:.4f}\n"
                f"Titulo: {hit.title}\n"
                f"Texto: {chunk_text or '(vacio)'}"
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
