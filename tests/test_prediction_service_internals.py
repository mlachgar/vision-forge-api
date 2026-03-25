from __future__ import annotations

from types import SimpleNamespace

import torch

from vision_forge_api.config.schema import TagPrompt, TagSet
from vision_forge_api.predict.service import Prediction, PredictionService


class _CatalogStub:
    def canonical_tags(self):
        return ("cat", "dog")

    def prompts_for_tag(self, tag: str):
        if tag == "cat":
            return (TagPrompt(template="a {tag}", weight=1.0),)
        return ()

    def list_tag_sets(self):
        return (
            TagSet(name="animals", canonical_tags=("cat", "dog")),
            TagSet(name="other", canonical_tags=("dog",)),
        )


class _StoreStub:
    def __init__(self, vectors=None, metadata=None):
        self._vectors = vectors or {}
        self._metadata = metadata or {}
        self.persist_calls = []

    def load(self):
        return dict(self._vectors)

    def load_metadata(self):
        return dict(self._metadata)

    def persist(self, payload, *, model_id, format_version):
        self.persist_calls.append((payload, model_id, format_version))
        self._vectors = dict(payload)


class _SiglipStub:
    def __init__(self):
        self.model_id = "model-1"
        self.device = torch.device("cpu")
        self.encode_images_calls = 0

    def encode_texts(self, texts):
        rows = []
        for i, _ in enumerate(texts):
            rows.append(torch.tensor([1.0 + i, 2.0 + i], dtype=torch.float32))
        return torch.stack(rows, dim=0)

    def encode_image(self, _image):
        return self.encode_images((_image,))[0]

    def encode_images(self, images):
        self.encode_images_calls += 1
        rows = []
        for i, _ in enumerate(images):
            rows.append(torch.tensor([1.0 + i, 0.5 + i], dtype=torch.float32))
        return torch.stack(rows, dim=0)


def _service() -> PredictionService:
    svc = object.__new__(PredictionService)
    svc._catalog = _CatalogStub()
    svc._siglip = _SiglipStub()
    svc._store = _StoreStub(
        vectors={"cat": (0.0, 1.0)},
        metadata={"format_version": 2, "model_id": "model-1"},
    )
    svc._vectors = {"cat": torch.tensor([0.0, 1.0]), "dog": torch.tensor([1.0, 0.0])}
    svc._prompt_vectors = {
        "cat": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "dog": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    }
    svc._canonical_primary_set = {"cat": "animals", "dog": "animals"}
    return svc


def test_vector_cache_and_prompt_helpers() -> None:
    svc = _service()

    cache = svc._build_vector_cache()
    assert set(cache.keys()) == {"cat", "dog"}

    prompt_cache = svc._build_prompt_vector_cache()
    assert set(prompt_cache.keys()) == {"cat", "dog"}

    primary_index = svc._build_primary_set_index()
    assert primary_index["cat"] == "animals"


def test_candidate_preparation_and_scoring_paths() -> None:
    svc = _service()

    vectors, labels, is_extra, canonical = svc._prepare_candidates(
        canonical_tags=("cat", "cat", "dog"),
        extra_labels=("extra", "extra"),
    )
    assert labels == ["cat", "dog", "extra"]
    assert is_extra == [False, False, True]
    assert canonical == ["cat", "dog"]

    image_vec = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    scores = svc._score_candidates(image_vec, vectors)
    assert len(scores) == 3

    svc._rerank_top_canonical(scores, canonical, image_vec, limit=1)

    built = svc._build_predictions(labels, is_extra, scores, min_score=-1.0)
    assert built
    assert built[0].score >= built[-1].score


def test_build_predictions_normalizes_scores_and_filters_threshold() -> None:
    svc = _service()

    built = svc._build_predictions(
        all_labels=["low", "mid", "high"],
        all_is_extra=[False, False, False],
        score_values=[-1.0, 0.0, 1.0],
        min_score=0.5,
    )

    assert [item.canonical_tag for item in built] == ["high", "mid"]
    assert [item.score for item in built] == [1.0, 0.5]


def test_set_balancing_and_misc_helpers() -> None:
    svc = _service()

    ranked = [
        Prediction("cat", 0.9, False),
        Prediction("dog", 0.89, False),
        Prediction("free", 0.88, True),
    ]
    assert svc._crosses_multiple_sets(ranked) is False

    svc._canonical_primary_set["dog"] = "other"
    assert svc._crosses_multiple_sets(ranked) is True

    balanced = svc._balance_results_by_set(ranked, limit=2)
    assert len(balanced) == 3

    assert svc.embedding_for_tag("cat") is not None
    assert svc.embedding_for_tag("missing") is None

    embedded = svc.embed_custom_label("tree")
    assert embedded.shape[-1] == 2


def test_score_image_full_flow() -> None:
    svc = _service()
    out = svc.score_image(
        image=SimpleNamespace(),
        canonical_tags=("cat", "dog"),
        extra_labels=("bonus",),
        min_score=-1.0,
        limit=2,
    )
    assert len(out) == 2

    empty = svc.score_image(
        image=SimpleNamespace(),
        canonical_tags=(),
        extra_labels=(),
        min_score=-1.0,
        limit=5,
    )
    assert empty == []


def test_score_images_uses_batched_encoding() -> None:
    svc = _service()
    out = svc.score_images(
        images=(SimpleNamespace(), SimpleNamespace()),
        canonical_tags=("cat", "dog"),
        extra_labels=("bonus",),
        min_score=-1.0,
        limit=2,
    )

    assert len(out) == 2
    assert svc._siglip.encode_images_calls == 1
