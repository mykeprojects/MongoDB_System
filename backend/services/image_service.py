from pathlib import Path

from backend.models.rag_models import ImageMatch
from backend.services.config_service import IMAGES_DIR


class ImageService:
    def __init__(self, images_dir: Path = IMAGES_DIR):
        self.images_dir = images_dir

    def resolve_public_path(self, image_path: str | None) -> str | None:
        if not image_path:
            return None
        normalized = image_path.replace("\\", "/").strip().lstrip("/")
        if normalized.startswith("data/images/"):
            return normalized
        return f"data/images/{Path(normalized).name}"

    def local_file_exists(self, image_path: str | None) -> bool:
        public_path = self.resolve_public_path(image_path)
        if not public_path:
            return False
        return (self.images_dir / Path(public_path).name).exists()

    def describe_image_path(self, image_path: str | None) -> str:
        if not image_path:
            return ""
        stem = Path(image_path).stem
        return stem.replace("_", " ").replace("-", " ").strip()

    def find_best_local_image(self, query: str, fallback: str | None = None) -> ImageMatch | None:
        files = [path for path in self.images_dir.glob("*") if path.is_file()]
        if not files:
            return None

        query_tokens = self._tokens(query)
        best: ImageMatch | None = None

        for file_path in files:
            title = self.describe_image_path(file_path.name)
            title_tokens = self._tokens(title)
            overlap = len(query_tokens & title_tokens)
            score = overlap / max(len(query_tokens | title_tokens), 1)
            match = ImageMatch(path=f"data/images/{file_path.name}", title=title, score=score)
            if best is None or match.score > best.score:
                best = match

        if best and best.score > 0:
            return best

        if fallback and self.local_file_exists(fallback):
            return ImageMatch(
                path=self.resolve_public_path(fallback) or fallback,
                title=self.describe_image_path(fallback),
                score=1.0,
            )

        first = files[0]
        return ImageMatch(path=f"data/images/{first.name}", title=self.describe_image_path(first.name), score=0.0)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        normalized = value.lower().replace("_", " ").replace("-", " ")
        return {token for token in normalized.split() if len(token) > 2}
