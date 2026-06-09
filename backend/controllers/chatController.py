import logging
from flask import Blueprint, current_app, jsonify, request

from backend.models.chat_dto import ChatRequestDTO

logger = logging.getLogger(__name__)

chat_blueprint = Blueprint("chat", __name__, url_prefix="/api")


@chat_blueprint.post("/chat")
def chat():
    try:
        logger.debug(f"Received chat request: {request.get_json(silent=True)}")
        dto = ChatRequestDTO.from_payload(request.get_json(silent=True))
        logger.debug(f"Created DTO: message={dto.message}, image_path={dto.image_path}")
        
        response = current_app.config["CHAT_SERVICE"].handle_chat(dto)
        logger.debug(f"Chat service returned: {response}")
        return jsonify(response.to_dict()), 200
    except ValueError as exc:
        logger.warning(f"Validation error: {exc}")
        return jsonify({"message": str(exc)}), 400
    except Exception as exc:
        logger.exception("Error procesando /api/chat")
        return jsonify({"message": f"Error interno del RAG: {exc}"}), 500


@chat_blueprint.get("/health")
def health():
    database = current_app.config["DATABASE_SERVICE"]
    config = current_app.config["APP_CONFIG"]
    try:
        database.ping()
        mongo = "ok"
    except Exception as exc:
        mongo = f"error: {exc}"

    return jsonify(
        {
            "status": "ok",
            "mongo": mongo,
            "groq": "configured" if config.groq_api_key else "missing_api_key",
            "groqModel": config.groq_model,
            "retrievalLimit": config.retrieval_limit,
            "retrievalStrategy": config.retrieval_strategy,
            "vectorIndex": config.vector_index_name,
        }
    ), 200
