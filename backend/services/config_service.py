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
    groq_api_key: str | None
    groq_model: str
    cors_origins: str
    retrieval_limit: int
    retrieval_strategy: str


def load_config() -> AppConfig:
    load_dotenv(BASE_DIR / ".env")

    return AppConfig(
        mongo_uri=os.getenv("MONGO_URI"),
        database_name=os.getenv("DATABASE_NAME"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        cors_origins=os.getenv("CORS_ORIGINS", "*"),
        retrieval_limit=int(os.getenv("RAG_RETRIEVAL_LIMIT", "4")),
        retrieval_strategy=os.getenv("RAG_CHUNK_STRATEGY", "semantico"),
    )
