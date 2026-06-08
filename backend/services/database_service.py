from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from backend.services.config_service import AppConfig


class DatabaseService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client: MongoClient | None = None

    @property
    def client(self) -> MongoClient:
        if not self.config.mongo_uri:
            raise RuntimeError("MONGO_URI no esta configurado en .env.")
        if self._client is None:
            self._client = MongoClient(self.config.mongo_uri, serverSelectionTimeoutMS=5000)
        return self._client

    @property
    def db(self) -> Database[dict[str, Any]]:
        if not self.config.database_name:
            raise RuntimeError("DATABASE_NAME no esta configurado en .env.")
        return self.client[self.config.database_name]

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
