import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"


@dataclass(frozen=True)
class AppConfig:
    mongo_uri: str | None
    database_name: str | None
    embedding_model: str
    embedding_dimensions: int
    clip_model: str
    clip_embedding_dimensions: int
    groq_api_key: str | None
    groq_model: str
    cors_origins: str
    retrieval_limit: int
    retrieval_strategy: str
    vector_index_name: str
    multimodal_text_index: str
    multimodal_clip_text_index: str
    multimodal_clip_image_index: str


def load_config() -> AppConfig:
    load_dotenv(BASE_DIR / ".env")

    return AppConfig(
        mongo_uri=os.getenv("MONGO_URI"),
        database_name=os.getenv("DATABASE_NAME"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "384")),
        clip_model=os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32"),
        clip_embedding_dimensions=int(os.getenv("CLIP_EMBEDDING_DIMENSIONS", "512")),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        cors_origins=os.getenv("CORS_ORIGINS", "*"),
        retrieval_limit=int(os.getenv("RAG_RETRIEVAL_LIMIT", "5")),
        retrieval_strategy=os.getenv("RAG_CHUNK_STRATEGY", "semantico"),
        vector_index_name=os.getenv("MONGO_VECTOR_INDEX", "embedding_vector_index"),
        multimodal_text_index=os.getenv("MONGO_MULTIMODAL_TEXT_INDEX", "idx_text_minilm_384"),
        multimodal_clip_text_index=os.getenv(
            "MONGO_MULTIMODAL_CLIP_TEXT_INDEX", "idx_clip_text_512"
        ),
        multimodal_clip_image_index=os.getenv(
            "MONGO_MULTIMODAL_CLIP_IMAGE_INDEX", "idx_clip_image_512"
        ),
    )
