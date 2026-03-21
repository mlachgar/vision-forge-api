"""SigLIP encoder helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

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
        self.processor = SiglipProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side="right",
        )
        # The transformers typing stubs for SigLIP model loading are not precise.
        self.model: Any = SiglipModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        torch.nn.Module.to(self.model, self.device)
        self.model.eval()
        logger.debug("Loaded SigLIP model %s on %s", model_id, self.device)

    def encode_image(self, image: Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
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
        return F.normalize(features, dim=-1)
