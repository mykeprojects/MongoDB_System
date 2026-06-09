import logging
from flask import Flask, send_from_directory
from flask_cors import CORS

from backend.controllers.chatController import chat_blueprint
from backend.services.chat_service import ChatService
from backend.services.config_service import DATA_DIR, load_config
from backend.services.database_service import DatabaseService
from backend.services.clip_embedding_service import ClipEmbeddingService
from backend.services.embedding_service import EmbeddingService
from backend.services.image_service import ImageService
from backend.services.llm_service import LLMService
from backend.services.multimodal_retrieval_service import MultimodalRetrievalService
from backend.services.retrieval_service import RetrievalService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    config = load_config()

    logger.info("Initializing Flask app with config...")
    origins = config.cors_origins if config.cors_origins == "*" else config.cors_origins.split(",")
    CORS(app, resources={r"/api/*": {"origins": origins}, r"/data/*": {"origins": origins}}, supports_credentials=True)

    logger.info("Initializing services...")
    try:
        #cada que ejecutamos un servicio logueamos para verificar su actividad
        database = DatabaseService(config)
        logger.info("✓ DatabaseService initialized")
        
        embeddings = EmbeddingService(config.embedding_model)
        logger.info("✓ EmbeddingService initialized")

        clip_embeddings = ClipEmbeddingService(config.clip_model)
        logger.info("✓ ClipEmbeddingService initialized")
        
        retrieval = RetrievalService(database, embeddings, config)
        logger.info("✓ RetrievalService initialized")

        multimodal = MultimodalRetrievalService(database, embeddings, clip_embeddings, config)
        logger.info("✓ MultimodalRetrievalService initialized")
        
        images = ImageService()
        logger.info("✓ ImageService initialized")
        
        llm = LLMService(config)
        logger.info("✓ LLMService initialized")
        
        chat_service = ChatService(config, retrieval, multimodal, llm, images)
        logger.info("✓ ChatService initialized")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}", exc_info=True)
        raise

    app.config["APP_CONFIG"] = config
    app.config["DATABASE_SERVICE"] = database
    app.config["CHAT_SERVICE"] = chat_service

    app.register_blueprint(chat_blueprint)

    @app.get("/")
    def root():
        return {
            "message": "RAG backend listo",
            "endpoints": ["/api/chat", "/api/health"],
        }

    @app.get("/data/images/<path:filename>")
    def data_images(filename: str):
        return send_from_directory(DATA_DIR / "images", filename)

    @app.teardown_appcontext
    def close_database(_exception=None):
        # The client is lazy and reusable; keep it open during the app lifetime.
        return None

    logger.info("✓ Flask app configured successfully on port 8000")
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
