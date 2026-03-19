"""Prediction utilities and embedding management."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL.Image import Image
import torch
import torch.nn.functional as F

from ..catalog.service import TagCatalog
from ..config.schema import TagPrompt
from ..embeddings.store import EmbeddingStore
from ..siglip.service import SiglipService


DEFAULT_EXTRA_PROMPT = "An image of {tag}."


@dataclass(frozen=True)
class Prediction:
    canonical_tag: str
    score: float
    is_extra: bool


class PredictionService:
    """Scores images against canonical and custom tag embeddings."""

    def __init__(self, catalog: TagCatalog, siglip: SiglipService, embeddings_dir: Path) -> None:
        self._catalog = catalog
        self._siglip = siglip
        self._store = EmbeddingStore(embeddings_dir)
        self._vectors: dict[str, torch.Tensor] = self._build_vector_cache()

    def _build_vector_cache(self) -> dict[str, torch.Tensor]:
        raw = self._store.load()
        missing = tuple(tag for tag in self._catalog.canonical_tags() if tag not in raw)
        if missing:
            computed = self._compute_missing_embeddings(missing)
            raw.update(computed)
        return {
            tag: torch.tensor(vector, device=self._siglip.device)
            for tag, vector in raw.items()
        }

    def _render_prompt(self, prompt: TagPrompt, tag: str) -> str:
        try:
            return prompt.template.format(tag=tag)
        except Exception:
            return prompt.template

    def _compute_missing_embeddings(self, tags: Iterable[str]) -> dict[str, tuple[float, ...]]:
        computed: dict[str, tuple[float, ...]] = {}
        for tag in tags:
            prompts = self._catalog.prompts_for_tag(tag)
            if not prompts:
                prompts = (TagPrompt(template=DEFAULT_EXTRA_PROMPT, weight=1.0),)
            texts: list[str] = []
            weights: list[float] = []
            for prompt in prompts:
                prompt_weight = float(prompt.weight)
                if prompt_weight <= 0:
                    continue
                texts.append(self._render_prompt(prompt, tag))
                weights.append(prompt_weight)
            if not texts:
                texts = [DEFAULT_EXTRA_PROMPT.format(tag=tag)]
                weights = [1.0]
            embeddings = self._siglip.encode_texts(texts)
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self._siglip.device)
            averaged = F.normalize(
                (embeddings * weight_tensor.unsqueeze(1)).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6),
                dim=-1,
            )
            computed[tag] = tuple(float(val) for val in averaged.cpu().tolist())
        return computed

    def embedding_for_tag(self, tag: str) -> torch.Tensor | None:
        return self._vectors.get(tag)

    def embed_custom_label(self, label: str) -> torch.Tensor:
        prompt = DEFAULT_EXTRA_PROMPT.format(tag=label)
        return self._siglip.encode_texts((prompt,))[0]

    def score_image(
        self,
        image: Image,
        canonical_tags: Sequence[str],
        extra_labels: Sequence[str],
        min_score: float,
        limit: int,
    ) -> list[Prediction]:
        image_vector = self._siglip.encode_image(image)
        canonical_keys = tuple(OrderedDict.fromkeys(canonical_tags))
        canonical_vectors: list[torch.Tensor] = []
        canonical_sources: list[str] = []
        for key in canonical_keys:
            vector = self.embedding_for_tag(key)
            if vector is None:
                continue
            canonical_vectors.append(vector)
            canonical_sources.append(key)

        extra_keys = tuple(OrderedDict.fromkeys(extra_labels))
        extra_vectors: list[torch.Tensor] = []
        for label in extra_keys:
            extra_vectors.append(self.embed_custom_label(label))

        all_vectors = canonical_vectors + extra_vectors
        all_labels = canonical_sources + list(extra_keys)
        all_is_extra = [False] * len(canonical_vectors) + [True] * len(extra_keys)
        if not all_vectors:
            return []
        candidate_tensor = torch.stack(all_vectors, dim=0)
        scores_tensor = torch.matmul(candidate_tensor, image_vector.T).squeeze(-1)
        results: list[Prediction] = []
        for label, is_extra, score_value in zip(all_labels, all_is_extra, scores_tensor.tolist()):
            if score_value < min_score:
                continue
            results.append(Prediction(canonical_tag=label, score=score_value, is_extra=is_extra))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:max(1, limit)]
