"""Prediction utilities and embedding management."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image as PILImage
from PIL.Image import Image
import torch
import torch.nn.functional as F

from ..catalog.service import TagCatalog
from ..config.schema import TagPrompt
from ..embeddings.store import EmbeddingStore, TEXT_EMBEDDING_FORMAT_VERSION
from ..siglip.service import SiglipService


DEFAULT_EXTRA_PROMPT = "An image of {tag}."
RERANK_POOL_MIN = 100
RERANK_POOL_MULTIPLIER = 20
SET_BALANCE_PENALTY = 0.01


@dataclass(frozen=True)
class Prediction:
    canonical_tag: str
    score: float
    is_extra: bool


class PredictionService:
    """Scores images against canonical and custom tag embeddings."""

    def __init__(
        self, catalog: TagCatalog, siglip: SiglipService, embeddings_dir: Path
    ) -> None:
        self._catalog = catalog
        self._siglip = siglip
        self._store = EmbeddingStore(embeddings_dir)
        self._vectors: dict[str, torch.Tensor] = self._build_vector_cache()
        self._prompt_vectors: dict[str, torch.Tensor] = {}
        self._canonical_primary_set: dict[str, str] = self._build_primary_set_index()

    def _build_vector_cache(self) -> dict[str, torch.Tensor]:
        raw = self._store.load()
        metadata = self._store.load_metadata()
        format_version = int(metadata.get("format_version", 0))
        model_id = str(metadata.get("model_id", "")).strip()
        reset_vectors = (
            format_version != TEXT_EMBEDDING_FORMAT_VERSION
            or model_id != self._siglip.model_id
        )
        if reset_vectors:
            raw = {}
        missing = tuple(tag for tag in self._catalog.canonical_tags() if tag not in raw)
        if missing:
            computed = self._compute_missing_embeddings(missing)
            raw.update(computed)
            self._store.persist(
                raw,
                model_id=self._siglip.model_id,
                format_version=TEXT_EMBEDDING_FORMAT_VERSION,
            )
        return {
            tag: torch.tensor(vector, device=self._siglip.device)
            for tag, vector in raw.items()
        }

    def _render_prompt(self, prompt: TagPrompt, tag: str) -> str:
        try:
            return prompt.template.format(tag=tag)
        except Exception:
            return prompt.template

    def _compute_missing_embeddings(
        self, tags: Iterable[str]
    ) -> dict[str, tuple[float, ...]]:
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
            weight_tensor = torch.tensor(
                weights, dtype=torch.float32, device=self._siglip.device
            )
            averaged = F.normalize(
                (embeddings * weight_tensor.unsqueeze(1)).sum(dim=0)
                / weight_tensor.sum().clamp_min(1e-6),
                dim=-1,
            )
            computed[tag] = tuple(float(val) for val in averaged.cpu().tolist())
        return computed

    def _build_prompt_vector_cache(self) -> dict[str, torch.Tensor]:
        cache: dict[str, torch.Tensor] = {}
        for tag in self._catalog.canonical_tags():
            cache[tag] = self._get_prompt_vectors(tag)
        return cache

    def warmup(self) -> None:
        """Prime prompt caches and a tiny image forward pass for lower first-request latency."""
        self._build_prompt_vector_cache()
        warmup_image = PILImage.new("RGB", (224, 224), color=(0, 0, 0))
        self._siglip.encode_image(warmup_image)

    def build_caption(
        self, predictions: Sequence[Prediction], max_tags: int = 3
    ) -> str | None:
        labels = [
            prediction.canonical_tag
            for prediction in predictions
            if prediction.canonical_tag.strip()
        ][: max(1, max_tags)]
        if not labels:
            return None
        if len(labels) == 1:
            body = labels[0]
        elif len(labels) == 2:
            body = f"{labels[0]} and {labels[1]}"
        else:
            body = ", ".join(labels[:-1]) + f", and {labels[-1]}"
        return f"An image showing {body}."

    def _get_prompt_vectors(self, tag: str) -> torch.Tensor:
        prompt_vectors = self._prompt_vectors.get(tag)
        if prompt_vectors is not None:
            return prompt_vectors

        prompts = self._catalog.prompts_for_tag(tag)
        texts: list[str] = []
        for prompt in prompts:
            if float(prompt.weight) <= 0:
                continue
            texts.append(self._render_prompt(prompt, tag))
        if not texts:
            texts = [DEFAULT_EXTRA_PROMPT.format(tag=tag)]
        prompt_vectors = self._siglip.encode_texts(texts)
        self._prompt_vectors[tag] = prompt_vectors
        return prompt_vectors

    def _build_primary_set_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for tag_set in self._catalog.list_tag_sets():
            for tag in tag_set.canonical_tags:
                index.setdefault(tag, tag_set.name)
        return index

    def _adjusted_set_score(
        self, item: Prediction, set_counts: dict[str, int]
    ) -> float:
        if item.is_extra:
            return item.score
        set_name = self._canonical_primary_set.get(item.canonical_tag)
        penalty = SET_BALANCE_PENALTY * set_counts.get(set_name or "", 0)
        return item.score - penalty

    def _pick_best_candidate_index(
        self, remaining: list[Prediction], set_counts: dict[str, int]
    ) -> int:
        best_idx = 0
        best_adjusted = float("-inf")
        best_raw = float("-inf")
        for idx, item in enumerate(remaining):
            adjusted = self._adjusted_set_score(item, set_counts)
            if adjusted > best_adjusted or (
                adjusted == best_adjusted and item.score > best_raw
            ):
                best_idx = idx
                best_adjusted = adjusted
                best_raw = item.score
        return best_idx

    def _register_selected_set(
        self, selected: Prediction, set_counts: dict[str, int]
    ) -> None:
        if selected.is_extra:
            return
        set_name = self._canonical_primary_set.get(selected.canonical_tag)
        if set_name:
            set_counts[set_name] = set_counts.get(set_name, 0) + 1

    def _balance_results_by_set(
        self, ranked: list[Prediction], limit: int
    ) -> list[Prediction]:
        max_items = max(1, limit)
        remaining = list(ranked)
        selected: list[Prediction] = []
        set_counts: dict[str, int] = {}
        while remaining and len(selected) < max_items:
            best_idx = self._pick_best_candidate_index(remaining, set_counts)
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            self._register_selected_set(chosen, set_counts)
        selected.extend(remaining)
        return selected

    def _prepare_candidates(
        self, canonical_tags: Sequence[str], extra_labels: Sequence[str]
    ) -> tuple[list[torch.Tensor], list[str], list[bool], list[str]]:
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
        extra_vectors = [self.embed_custom_label(label) for label in extra_keys]
        all_vectors = canonical_vectors + extra_vectors
        all_labels = canonical_sources + list(extra_keys)
        all_is_extra = [False] * len(canonical_vectors) + [True] * len(extra_keys)
        return all_vectors, all_labels, all_is_extra, canonical_sources

    @staticmethod
    def _score_candidates(
        image_vector: torch.Tensor, all_vectors: list[torch.Tensor]
    ) -> list[float]:
        candidate_tensor = torch.stack(all_vectors, dim=0)
        image_vector_1d = image_vector[0] if image_vector.ndim > 1 else image_vector
        image_column = image_vector_1d.unsqueeze(-1)
        scores_tensor = torch.matmul(candidate_tensor, image_column).squeeze(-1)
        return scores_tensor.tolist()

    def _rerank_top_canonical(
        self,
        score_values: list[float],
        canonical_sources: list[str],
        image_vector: torch.Tensor,
        limit: int,
    ) -> None:
        if not canonical_sources:
            return
        ranked_idx = sorted(
            range(len(canonical_sources)),
            key=lambda idx: score_values[idx],
            reverse=True,
        )
        rerank_pool = min(
            len(ranked_idx), max(limit * RERANK_POOL_MULTIPLIER, RERANK_POOL_MIN)
        )
        for idx in ranked_idx[:rerank_pool]:
            label = canonical_sources[idx]
            prompt_vectors = self._get_prompt_vectors(label)
            if prompt_vectors.numel() == 0:
                continue
            image_vector_1d = image_vector[0] if image_vector.ndim > 1 else image_vector
            image_column = image_vector_1d.unsqueeze(-1)
            prompt_scores = torch.matmul(prompt_vectors, image_column).squeeze(-1)
            score_values[idx] = float(torch.max(prompt_scores).item())

    @staticmethod
    def _normalize_score(score_value: float) -> float:
        # SigLIP returns cosine similarities in [-1, 1]. Expose a normalized
        # confidence-like score so callers can use familiar 0..1 thresholds.
        return min(1.0, max(0.0, (score_value + 1.0) / 2.0))

    def _build_predictions(
        self,
        all_labels: list[str],
        all_is_extra: list[bool],
        score_values: list[float],
        min_score: float,
    ) -> list[Prediction]:
        results: list[Prediction] = []
        for label, is_extra, score_value in zip(all_labels, all_is_extra, score_values):
            normalized_score = self._normalize_score(score_value)
            if normalized_score < min_score:
                continue
            results.append(
                Prediction(
                    canonical_tag=label,
                    score=normalized_score,
                    is_extra=is_extra,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _crosses_multiple_sets(self, results: list[Prediction]) -> bool:
        result_sets = {
            self._canonical_primary_set.get(item.canonical_tag)
            for item in results
            if not item.is_extra and self._canonical_primary_set.get(item.canonical_tag)
        }
        return len(result_sets) > 1

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
        all_vectors, all_labels, all_is_extra, canonical_sources = (
            self._prepare_candidates(
                canonical_tags=canonical_tags, extra_labels=extra_labels
            )
        )
        if not all_vectors:
            return []
        score_values = self._score_candidates(
            image_vector=image_vector, all_vectors=all_vectors
        )
        self._rerank_top_canonical(
            score_values=score_values,
            canonical_sources=canonical_sources,
            image_vector=image_vector,
            limit=limit,
        )
        results = self._build_predictions(
            all_labels=all_labels,
            all_is_extra=all_is_extra,
            score_values=score_values,
            min_score=min_score,
        )
        if self._crosses_multiple_sets(results):
            results = self._balance_results_by_set(results, limit)
        return results[: max(1, limit)]

    def score_images(
        self,
        images: Sequence[Image],
        canonical_tags: Sequence[str],
        extra_labels: Sequence[str],
        min_score: float,
        limit: int,
    ) -> list[list[Prediction]]:
        if not images:
            return []
        all_vectors, all_labels, all_is_extra, canonical_sources = (
            self._prepare_candidates(
                canonical_tags=canonical_tags, extra_labels=extra_labels
            )
        )
        if not all_vectors:
            return [[] for _ in images]

        image_vectors = self._siglip.encode_images(images)
        candidate_tensor = torch.stack(all_vectors, dim=0)
        score_matrix = torch.matmul(image_vectors, candidate_tensor.T)

        results: list[list[Prediction]] = []
        for image_idx, image_vector in enumerate(image_vectors):
            score_values = score_matrix[image_idx].tolist()
            self._rerank_top_canonical(
                score_values=score_values,
                canonical_sources=canonical_sources,
                image_vector=image_vector,
                limit=limit,
            )
            item_results = self._build_predictions(
                all_labels=all_labels,
                all_is_extra=all_is_extra,
                score_values=score_values,
                min_score=min_score,
            )
            if self._crosses_multiple_sets(item_results):
                item_results = self._balance_results_by_set(item_results, limit)
            results.append(item_results[: max(1, limit)])
        return results
