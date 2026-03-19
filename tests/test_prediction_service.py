from __future__ import annotations

import torch

from vision_forge_api.predict.service import Prediction, PredictionService


class _SiglipStub:
    def encode_image(self, _):
        return torch.tensor([[1.0]], dtype=torch.float32)


def test_balance_results_by_set_penalizes_dominant_set() -> None:
    service = object.__new__(PredictionService)
    service._canonical_primary_set = {
        "a1": "set_a",
        "a2": "set_a",
        "b1": "set_b",
    }

    ranked = [
        Prediction(canonical_tag="a1", score=0.90, is_extra=False),
        Prediction(canonical_tag="a2", score=0.885, is_extra=False),
        Prediction(canonical_tag="b1", score=0.88, is_extra=False),
    ]

    balanced = service._balance_results_by_set(ranked, limit=3)

    assert [item.canonical_tag for item in balanced[:3]] == ["a1", "b1", "a2"]


def test_score_image_returns_empty_when_no_candidates() -> None:
    service = object.__new__(PredictionService)
    service._siglip = _SiglipStub()
    service._prepare_candidates = lambda canonical_tags, extra_labels: ([], [], [], [])

    output = service.score_image(
        image=None,
        canonical_tags=("a",),
        extra_labels=("b",),
        min_score=-1.0,
        limit=5,
    )

    assert output == []


def test_score_image_uses_balancing_when_multiple_sets() -> None:
    service = object.__new__(PredictionService)
    service._siglip = _SiglipStub()
    service._prepare_candidates = lambda canonical_tags, extra_labels: (
        [torch.tensor([1.0]), torch.tensor([1.0])],
        ["a1", "b1"],
        [False, False],
        ["a1", "b1"],
    )
    service._score_candidates = lambda image_vector, all_vectors: [0.9, 0.8]
    service._rerank_top_canonical = (
        lambda score_values, canonical_sources, image_vector, limit: None
    )
    service._build_predictions = (
        lambda all_labels, all_is_extra, score_values, min_score: [
            Prediction(canonical_tag="a1", score=0.9, is_extra=False),
            Prediction(canonical_tag="b1", score=0.8, is_extra=False),
        ]
    )
    service._crosses_multiple_sets = lambda results: True
    service._balance_results_by_set = lambda ranked, limit: [
        Prediction(canonical_tag="b1", score=0.8, is_extra=False),
        Prediction(canonical_tag="a1", score=0.9, is_extra=False),
    ]

    output = service.score_image(
        image=None,
        canonical_tags=("a1",),
        extra_labels=(),
        min_score=-1.0,
        limit=1,
    )

    assert len(output) == 1
    assert output[0].canonical_tag == "b1"
