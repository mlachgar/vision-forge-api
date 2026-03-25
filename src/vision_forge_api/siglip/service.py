"""SigLIP encoder helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence, cast

from PIL.Image import Image
import torch
import torch.nn.functional as F
from transformers import SiglipModel, SiglipProcessor

logger = logging.getLogger(__name__)


def _resolve_device(preferred: str | None = None) -> torch.device:
    candidate = (preferred or "cpu").lower()
    if candidate in {"gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SiglipService:
    """Wraps the SigLIP processor/model for encoding."""

    def __init__(self, model_id: str, cache_dir: Path, device_hint: str | None = None):
        self.model_id = model_id
        self.device = _resolve_device(device_hint)
        self._cache_dir = cache_dir
        self.processor, self.model = self._load_model()

    @staticmethod
    def _load_flag(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _load_model(self) -> tuple[Any, Any]:
        processor = SiglipProcessor.from_pretrained(
            self.model_id,
            cache_dir=self._cache_dir,
            padding_side="right",
        )
        # The transformers typing stubs for SigLIP model loading are not precise.
        model: Any = SiglipModel.from_pretrained(
            self.model_id,
            cache_dir=self._cache_dir,
        )
        cast(Any, model).to(self.device)
        model.eval()
        logger.debug("Loaded SigLIP model %s on %s", self.model_id, self.device)
        return processor, model

    def preload(self) -> None:
        """Compatibility hook for build-time warmup scripts."""
        return None

    @staticmethod
    def _as_feature_tensor(features: Any) -> torch.Tensor:
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(features, attr, None)
            if value is not None:
                return value
        if isinstance(features, (tuple, list)) and features:
            return features[0]
        return features

    def encode_image(self, image: Image) -> torch.Tensor:
        return self.encode_images((image,))[0]

    def encode_images(self, images: Sequence[Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, 0), device=self.device)
        inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
        features = self._as_feature_tensor(features)
        return F.normalize(features, dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, device=self.device)
        # SigLIP is calibrated with fixed-length text inputs; dynamic padding degrades retrieval quality.
        inputs = self.processor(
            text=list(texts), padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = self._as_feature_tensor(features)
        return F.normalize(features, dim=-1)
