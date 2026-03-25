from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision_forge_api.api.services.predict import PreparedPredictionOptions
from vision_forge_api.api.services.predict_jobs import (
    ITEM_STATUS_DONE,
    JOB_STATUS_DONE,
    PredictJobItemResult,
    PredictJobRecord,
    PredictJobService,
    _QueuedItem,
)
from vision_forge_api.predict.service import Prediction


class _PredictionServiceStub:
    def __init__(self) -> None:
        self.calls = 0

    def score_images(self, *, images, canonical_tags, extra_labels, min_score, limit):
        self.calls += 1
        return [
            [Prediction(canonical_tag="cat", score=0.9, is_extra=False)] for _ in images
        ]


@pytest.mark.asyncio
async def test_process_batch_updates_job_state() -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    service._decode_image = lambda _payload: SimpleNamespace(width=1, height=1)
    options = PreparedPredictionOptions(
        canonical_tags=("cat",),
        extra_tags=(),
        selected_tag_sets=("animals",),
        resolved_profile="default",
        limit=1,
        min_score=0.0,
    )
    record = PredictJobRecord(job_id="job-1", total_items=2)
    record.items = [
        PredictJobItemResult(item_id="item-1", filename="one.jpg"),
        PredictJobItemResult(item_id="item-2", filename="two.jpg"),
    ]
    service._jobs[record.job_id] = record

    await service._process_batch(
        [
            _QueuedItem(
                job_id="job-1",
                item_id="item-1",
                filename="one.jpg",
                payload=b"fake-1",
                options_signature="sig",
                options=options,
            ),
            _QueuedItem(
                job_id="job-1",
                item_id="item-2",
                filename="two.jpg",
                payload=b"fake-2",
                options_signature="sig",
                options=options,
            ),
        ]
    )

    snapshot = service.get_job("job-1")
    assert snapshot.status == JOB_STATUS_DONE
    assert snapshot.completed_items == 2
    assert snapshot.failed_items == 0
    assert [item.status for item in snapshot.items] == [
        ITEM_STATUS_DONE,
        ITEM_STATUS_DONE,
    ]
    assert snapshot.items[0].tags[0][0] == "cat"
