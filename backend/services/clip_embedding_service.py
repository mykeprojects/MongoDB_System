import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ClipEmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from transformers import CLIPModel, CLIPProcessor

                    self._model = CLIPModel.from_pretrained(self.model_name)
                    self._processor = CLIPProcessor.from_pretrained(self.model_name)
                    self._model.eval()

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
            text_features = self.model.text_projection(pooled)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().astype(float)

    def encode_image_path(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.is_absolute() and not path.exists():
            from backend.services.config_service import IMAGES_DIR

            path = IMAGES_DIR / path.name
        with Image.open(path) as image:
            return self.encode_image(image)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
            image_features = self.model.visual_projection(pooled)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features[0].cpu().numpy().astype(float)
