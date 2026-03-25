from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from vision_forge_api.api.errors import (
    BadRequestError,
    NotFoundError,
    ServiceUnavailableError,
)
from vision_forge_api.api.services.predict import PreparedPredictionOptions
import vision_forge_api.api.services.predict_jobs as predict_jobs_mod
from vision_forge_api.api.services.predict_jobs import (
    ITEM_STATUS_DONE,
    ITEM_STATUS_FAILED,
    JOB_STATUS_CANCELED,
    JOB_STATUS_DONE,
    JOB_STATUS_FAILED,
    JOB_STATUS_PARTIAL,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    MAX_STORED_ITEMS,
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


class _UploadStub:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self._payload = payload

    async def read(self) -> bytes:
        await asyncio.sleep(0)
        return self._payload


def _options(*, include_caption: bool = False) -> PreparedPredictionOptions:
    return PreparedPredictionOptions(
        canonical_tags=("cat",),
        extra_tags=("bonus",),
        selected_tag_sets=("animals",),
        resolved_profile="default",
        limit=2,
        min_score=0.1,
        include_caption=include_caption,
    )


def test_process_batch_updates_job_state() -> None:
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

    service._process_batch(
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


def test_cleanup_retained_jobs_evicts_expired_terminal_jobs() -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)

    expired = PredictJobRecord(
        job_id="old",
        status=JOB_STATUS_DONE,
        total_items=1,
        completed_items=1,
        finished_at=datetime.now(timezone.utc) - timedelta(minutes=16),
    )
    active = PredictJobRecord(
        job_id="active",
        status="running",
        total_items=1,
        completed_items=0,
    )
    service._jobs = {"old": expired, "active": active}

    service._cleanup_retained_jobs(trim_to_capacity=True)

    assert "old" not in service._jobs
    assert "active" in service._jobs


class _AwaitableTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def __await__(self):
        if False:
            yield None
        return None


@pytest.mark.asyncio
async def test_start_and_stop_manage_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    created: list[str] = []

    def _create_task(coro, name=None):
        coro.close()
        created.append(name or "")
        return _AwaitableTask()

    monkeypatch.setattr(predict_jobs_mod.asyncio, "create_task", _create_task)

    service.start()
    service.start()
    assert created == ["predict-job-worker", "predict-job-reaper"]

    await service.stop()
    assert service._worker_task is None
    assert service._reaper_task is None


@pytest.mark.asyncio
async def test_submit_job_rejects_too_many_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    monkeypatch.setattr(predict_jobs_mod, "MAX_STORED_ITEMS", 0)

    with pytest.raises(ServiceUnavailableError):
        await service.submit_job(
            files=[_UploadStub("one.jpg", b"payload")],
            options=_options(),
        )


@pytest.mark.asyncio
async def test_stop_handles_missing_worker_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    service._reaper_task = _AwaitableTask()
    service._worker_task = None

    await service.stop()

    assert service._reaper_task is None
    assert service._worker_task is None


@pytest.mark.asyncio
async def test_submit_job_creates_snapshot_and_rejects_bad_inputs() -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)

    created = await service.submit_job(
        files=[_UploadStub("one.jpg", b"payload")],
        options=_options(),
    )

    assert created.total_items == 1
    assert created.status == JOB_STATUS_QUEUED
    assert created.items[0].filename == "one.jpg"
    assert service.get_job(created.job_id).items[0].status == "queued"

    with pytest.raises(BadRequestError):
        await service.submit_job(files=[], options=_options())

    with pytest.raises(BadRequestError):
        await service.submit_job(
            files=[_UploadStub(" ", b"payload")], options=_options()
        )

    with pytest.raises(BadRequestError):
        await service.submit_job(
            files=[_UploadStub("bad.jpg", b"")], options=_options()
        )


def test_get_cancel_and_internal_status_helpers() -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    running = PredictJobRecord(
        job_id="job-1",
        status=JOB_STATUS_RUNNING,
        total_items=1,
        completed_items=0,
        items=[PredictJobItemResult(item_id="item-1", filename="one.jpg")],
    )
    done = PredictJobRecord(
        job_id="job-2",
        status=JOB_STATUS_DONE,
        total_items=1,
        completed_items=1,
        finished_at=datetime.now(timezone.utc),
    )
    service._jobs = {running.job_id: running, done.job_id: done}

    snapshot = service.get_job("job-1")
    assert snapshot is not running
    assert snapshot.items[0] is not running.items[0]

    canceled = service.cancel_job("job-1")
    assert canceled.status == JOB_STATUS_CANCELED
    assert canceled.finished_at is not None

    assert service.cancel_job("job-2").status == JOB_STATUS_DONE

    assert service._job_status(PredictJobRecord(job_id="queued")) == JOB_STATUS_QUEUED
    assert (
        service._job_status(
            PredictJobRecord(job_id="running", total_items=2, completed_items=1)
        )
        == JOB_STATUS_RUNNING
    )
    assert (
        service._job_status(
            PredictJobRecord(
                job_id="partial",
                total_items=2,
                completed_items=2,
                failed_items=1,
            )
        )
        == JOB_STATUS_PARTIAL
    )
    assert (
        service._job_status(
            PredictJobRecord(
                job_id="failed",
                total_items=2,
                completed_items=2,
                failed_items=2,
            )
        )
        == JOB_STATUS_FAILED
    )
    assert service._job_status(done) == JOB_STATUS_DONE
    assert service._is_terminal(done) is True
    assert (
        service._is_terminal(PredictJobRecord(job_id="live", status=JOB_STATUS_RUNNING))
        is False
    )
    assert (
        service._job_status(
            PredictJobRecord(job_id="canceled", status=JOB_STATUS_CANCELED)
        )
        == JOB_STATUS_CANCELED
    )

    with pytest.raises(NotFoundError):
        service.get_job("missing")
    with pytest.raises(NotFoundError):
        service.cancel_job("missing")


def test_cleanup_retained_jobs_trims_terminal_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    monkeypatch.setattr(predict_jobs_mod, "MAX_STORED_ITEMS", 1)

    old = PredictJobRecord(
        job_id="old",
        status=JOB_STATUS_DONE,
        total_items=1,
        completed_items=1,
        finished_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    new = PredictJobRecord(
        job_id="new",
        status=JOB_STATUS_DONE,
        total_items=1,
        completed_items=1,
        finished_at=datetime.now(timezone.utc),
    )
    service._jobs = {old.job_id: old, new.job_id: new}

    service._cleanup_retained_jobs(trim_to_capacity=True)

    assert len(service._jobs) == 1
    assert "new" in service._jobs


def test_batch_helpers_and_finalize_paths() -> None:
    context = SimpleNamespace(
        prediction_service=SimpleNamespace(
            build_caption=lambda predictions: f"caption:{predictions[0].canonical_tag}"
        )
    )
    service = PredictJobService(context, batch_size=1, flush_interval_seconds=0.0)
    options = _options(include_caption=True)

    assert (
        service._options_signature(options)
        == "default|animals|bonus|2|0.100000|caption"
    )

    batch = [
        _QueuedItem(
            job_id="job-old",
            item_id="item-old",
            filename="old.jpg",
            payload=b"old",
            options_signature="sig-a",
            options=options,
        ),
        _QueuedItem(
            job_id="job-new",
            item_id="item-new",
            filename="new.jpg",
            payload=b"new",
            options_signature="sig-b",
            options=options,
        ),
    ]
    assert list(service._group_batch(batch).keys()) == ["sig-a", "sig-b"]

    old_record = PredictJobRecord(
        job_id="job-old",
        status=JOB_STATUS_QUEUED,
        total_items=1,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        items=[PredictJobItemResult(item_id="item-old", filename="old.jpg")],
    )
    new_record = PredictJobRecord(
        job_id="job-new",
        status=JOB_STATUS_QUEUED,
        total_items=1,
        created_at=datetime.now(timezone.utc),
        items=[PredictJobItemResult(item_id="item-new", filename="new.jpg")],
    )
    service._jobs = {old_record.job_id: old_record, new_record.job_id: new_record}
    pending = {
        "sig-a": [batch[0]],
        "sig-b": [batch[1]],
    }
    taken = service._take_pending_batch(pending)
    assert taken is not None
    assert taken[0].job_id == "job-old"

    service._queue.put_nowait(
        _QueuedItem(
            job_id="job-old",
            item_id="item-old",
            filename="old.jpg",
            payload=b"old",
            options_signature="sig-a",
            options=options,
        )
    )
    service._queue.put_nowait(None)
    drained: dict[str, list[_QueuedItem]] = defaultdict(list)
    service._drain_pending_queue(drained)
    assert service._stopping is True

    bad_record = PredictJobRecord(
        job_id="job-bad",
        total_items=1,
        items=[PredictJobItemResult(item_id="item-bad", filename="bad.jpg")],
    )
    service._jobs[bad_record.job_id] = bad_record
    prepared, staged_records, images = service._prepare_signature_batch(
        [
            _QueuedItem(
                job_id="job-bad",
                item_id="item-bad",
                filename="bad.jpg",
                payload=b"not-an-image",
                options_signature="sig-bad",
                options=options,
            )
        ]
    )
    assert prepared == options
    assert staged_records == []
    assert images == []
    assert bad_record.items[0].status == ITEM_STATUS_FAILED
    assert bad_record.status == JOB_STATUS_FAILED

    done_record = PredictJobRecord(
        job_id="job-done",
        total_items=1,
        items=[PredictJobItemResult(item_id="item-done", filename="done.jpg")],
    )
    service._jobs[done_record.job_id] = done_record
    service._apply_predictions(
        [(done_record, done_record.items[0])],
        [[Prediction(canonical_tag="cat", score=0.9, is_extra=False)]],
        include_caption=True,
    )
    assert done_record.items[0].status == ITEM_STATUS_DONE
    assert done_record.items[0].caption == "caption:cat"
    assert done_record.status == JOB_STATUS_DONE

    partial_record = PredictJobRecord(
        job_id="job-partial",
        status=JOB_STATUS_RUNNING,
        total_items=2,
        completed_items=2,
        failed_items=1,
    )
    service._finalize_job(partial_record)
    assert partial_record.status == JOB_STATUS_PARTIAL

    canceled_record = PredictJobRecord(
        job_id="job-canceled",
        status=JOB_STATUS_CANCELED,
        total_items=1,
        completed_items=1,
    )
    service._finalize_job(canceled_record)
    assert canceled_record.status == JOB_STATUS_CANCELED

    service._jobs[canceled_record.job_id] = canceled_record
    canceled_queued = _QueuedItem(
        job_id="job-canceled",
        item_id="item-canceled",
        filename="canceled.jpg",
        payload=b"payload",
        options_signature="sig-canceled",
        options=options,
    )
    canceled_record.items = [
        PredictJobItemResult(item_id="item-canceled", filename="canceled.jpg")
    ]
    prepared, staged_records, images = service._prepare_signature_batch(
        [canceled_queued]
    )
    assert prepared == options
    assert staged_records == []
    assert images == []


def test_queue_helpers_handle_empty_and_missing_items() -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    options = _options()

    assert service._prepare_signature_batch([]) == (None, [], [])
    assert service._take_pending_batch({}) is None

    empty_pending: dict[str, list[_QueuedItem]] = defaultdict(list)
    service._drain_pending_queue(empty_pending)
    assert empty_pending == {}

    service._process_batch([])
    service._process_signature_batch([])

    missing = _QueuedItem(
        job_id="missing",
        item_id="item",
        filename="missing.jpg",
        payload=b"payload",
        options_signature="sig",
        options=options,
    )
    service._jobs["missing"] = PredictJobRecord(
        job_id="missing",
        total_items=1,
        items=[],
    )
    prepared, staged_records, images = service._prepare_signature_batch([missing])
    assert prepared == options
    assert staged_records == []
    assert images == []


@pytest.mark.asyncio
async def test_submit_job_rejects_when_capacity_is_full() -> None:
    context = SimpleNamespace(prediction_service=_PredictionServiceStub())
    service = PredictJobService(context)
    service._jobs = {
        "running": PredictJobRecord(
            job_id="running",
            status="running",
            total_items=MAX_STORED_ITEMS,
            completed_items=0,
        )
    }

    with pytest.raises(ServiceUnavailableError):
        await service.submit_job(
            files=[_UploadStub("one.jpg", b"payload")],
            options=PreparedPredictionOptions(
                canonical_tags=("cat",),
                extra_tags=(),
                selected_tag_sets=("animals",),
                resolved_profile="default",
                limit=1,
                min_score=0.0,
            ),
        )
