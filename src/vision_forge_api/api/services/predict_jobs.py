"""In-memory prediction job queue and micro-batching."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from contextlib import suppress
from io import BytesIO
from typing import TYPE_CHECKING, Sequence
from uuid import uuid4

from PIL import Image
from fastapi import UploadFile

from ..errors import BadRequestError, NotFoundError, ServiceUnavailableError
from .predict import PreparedPredictionOptions

if TYPE_CHECKING:
    from ..context import AppContext


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_PARTIAL = "partial"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELED = "canceled"

ITEM_STATUS_QUEUED = "queued"
ITEM_STATUS_RUNNING = "running"
ITEM_STATUS_DONE = "done"
ITEM_STATUS_FAILED = "failed"

JOB_TTL_SECONDS = 15 * 60
MAX_STORED_ITEMS = 5000
CLEANUP_INTERVAL_SECONDS = 60.0
_TERMINAL_JOB_STATUSES = {
    JOB_STATUS_DONE,
    JOB_STATUS_PARTIAL,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELED,
}


@dataclass(slots=True)
class PredictJobItemResult:
    item_id: str
    filename: str
    status: str = ITEM_STATUS_QUEUED
    tags: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class PredictJobRecord:
    job_id: str
    status: str = JOB_STATUS_QUEUED
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    finished_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    items: list[PredictJobItemResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _QueuedItem:
    job_id: str
    item_id: str
    filename: str
    payload: bytes
    options_signature: str
    options: PreparedPredictionOptions


class PredictJobService:
    """Queues prediction requests and executes them in small batches."""

    def __init__(
        self,
        context: AppContext,
        *,
        batch_size: int = 8,
        flush_interval_seconds: float = 0.02,
    ) -> None:
        self._context = context
        self._batch_size = max(1, batch_size)
        self._flush_interval_seconds = max(0.0, flush_interval_seconds)
        self._queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
        self._jobs: dict[str, PredictJobRecord] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._reaper_task: asyncio.Task[None] | None = None
        self._stopping = False

    def start(self) -> None:
        if self._worker_task is not None or self._reaper_task is not None:
            return
        self._stopping = False
        self._worker_task = asyncio.create_task(
            self._worker(), name="predict-job-worker"
        )
        self._reaper_task = asyncio.create_task(
            self._reaper(), name="predict-job-reaper"
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None
        if self._worker_task is None:
            return
        await self._queue.put(None)
        self._worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None

    async def submit_job(
        self,
        *,
        files: Sequence[UploadFile],
        options: PreparedPredictionOptions,
    ) -> PredictJobRecord:
        if not files:
            raise BadRequestError("at least one file is required")

        requested_items = len(files)
        if requested_items > MAX_STORED_ITEMS:
            raise ServiceUnavailableError(
                "prediction job is too large, please submit fewer items"
            )

        payloads: list[tuple[str, bytes]] = []
        for upload in files:
            filename = (upload.filename or "").strip()
            if not filename:
                raise BadRequestError("each uploaded file must have a filename")
            payload = await upload.read()
            if not payload:
                raise BadRequestError(f"file '{filename}' is empty")
            payloads.append((filename, payload))

        self._cleanup_retained_jobs(trim_to_capacity=True)
        if self._retained_items_count() + requested_items > MAX_STORED_ITEMS:
            raise ServiceUnavailableError(
                "prediction job queue is at capacity, please retry later"
            )

        job_id = uuid4().hex
        record = PredictJobRecord(job_id=job_id, total_items=len(payloads))
        self._jobs[job_id] = record

        for filename, payload in payloads:
            item_id = uuid4().hex
            item = PredictJobItemResult(item_id=item_id, filename=filename)
            record.items.append(item)
            await self._queue.put(
                _QueuedItem(
                    job_id=job_id,
                    item_id=item_id,
                    filename=filename,
                    payload=payload,
                    options_signature=self._options_signature(options),
                    options=options,
                )
            )

        self._touch(record)
        return self._snapshot(record)

    def get_job(self, job_id: str) -> PredictJobRecord:
        self._cleanup_retained_jobs()
        record = self._jobs.get(job_id)
        if record is None:
            raise NotFoundError(f"prediction job '{job_id}' not found")
        return self._snapshot(record)

    def cancel_job(self, job_id: str) -> PredictJobRecord:
        self._cleanup_retained_jobs()
        record = self._jobs.get(job_id)
        if record is None:
            raise NotFoundError(f"prediction job '{job_id}' not found")
        if record.status not in {
            JOB_STATUS_DONE,
            JOB_STATUS_PARTIAL,
            JOB_STATUS_FAILED,
        }:
            record.status = JOB_STATUS_CANCELED
            self._mark_finished(record)
            self._touch(record)
            self._cleanup_retained_jobs(trim_to_capacity=True)
        return self._snapshot(record)

    @staticmethod
    def _options_signature(options: PreparedPredictionOptions) -> str:
        return "|".join(
            [
                options.resolved_profile,
                ",".join(options.selected_tag_sets),
                ",".join(options.extra_tags),
                str(options.limit),
                f"{options.min_score:.6f}",
            ]
        )

    @staticmethod
    def _touch(record: PredictJobRecord) -> None:
        record.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _snapshot(record: PredictJobRecord) -> PredictJobRecord:
        return PredictJobRecord(
            job_id=record.job_id,
            status=record.status,
            total_items=record.total_items,
            completed_items=record.completed_items,
            failed_items=record.failed_items,
            finished_at=record.finished_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
            items=[
                PredictJobItemResult(
                    item_id=item.item_id,
                    filename=item.filename,
                    status=item.status,
                    tags=list(item.tags),
                    error=item.error,
                )
                for item in record.items
            ],
        )

    def _job_status(self, record: PredictJobRecord) -> str:
        if record.status == JOB_STATUS_CANCELED:
            return JOB_STATUS_CANCELED
        if record.completed_items == record.total_items and record.total_items > 0:
            if record.failed_items == record.total_items:
                return JOB_STATUS_FAILED
            if record.failed_items > 0:
                return JOB_STATUS_PARTIAL
            return JOB_STATUS_DONE
        if record.completed_items > 0 or record.failed_items > 0:
            return JOB_STATUS_RUNNING
        return JOB_STATUS_QUEUED

    @staticmethod
    def _is_terminal(record: PredictJobRecord) -> bool:
        return record.status in _TERMINAL_JOB_STATUSES

    def _mark_finished(self, record: PredictJobRecord) -> None:
        if record.finished_at is None:
            record.finished_at = self._now()

    def _finalize_job(self, record: PredictJobRecord) -> None:
        if record.status == JOB_STATUS_CANCELED:
            return
        if record.completed_items < record.total_items:
            record.status = JOB_STATUS_RUNNING
        else:
            if record.failed_items == 0:
                record.status = JOB_STATUS_DONE
            elif record.failed_items == record.total_items:
                record.status = JOB_STATUS_FAILED
            else:
                record.status = JOB_STATUS_PARTIAL
            self._mark_finished(record)
        self._touch(record)
        if self._is_terminal(record):
            self._cleanup_retained_jobs(trim_to_capacity=True)

    def _mark_item_running(
        self, item: PredictJobItemResult, record: PredictJobRecord
    ) -> None:
        if record.status == JOB_STATUS_QUEUED:
            record.status = JOB_STATUS_RUNNING
        item.status = ITEM_STATUS_RUNNING
        self._touch(record)

    def _mark_item_done(
        self,
        item: PredictJobItemResult,
        record: PredictJobRecord,
        predictions: Sequence[tuple[str, float]],
    ) -> None:
        item.status = ITEM_STATUS_DONE
        item.tags = list(predictions)
        record.completed_items += 1
        self._finalize_job(record)

    def _mark_item_failed(
        self, item: PredictJobItemResult, record: PredictJobRecord, error: str
    ) -> None:
        item.status = ITEM_STATUS_FAILED
        item.error = error
        record.completed_items += 1
        record.failed_items += 1
        self._finalize_job(record)

    @staticmethod
    def _decode_image(payload: bytes) -> Image.Image:
        try:
            return Image.open(BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise BadRequestError("Unable to decode image payload") from exc

    def _retained_items_count(self) -> int:
        return sum(record.total_items for record in self._jobs.values())

    def _cleanup_retained_jobs(self, *, trim_to_capacity: bool = False) -> None:
        now = self._now()
        ttl = timedelta(seconds=JOB_TTL_SECONDS)

        expired_job_ids = [
            job_id
            for job_id, record in self._jobs.items()
            if self._is_terminal(record)
            and record.finished_at is not None
            and now - record.finished_at >= ttl
        ]
        for job_id in expired_job_ids:
            del self._jobs[job_id]

        if not trim_to_capacity:
            return

        retained_items = self._retained_items_count()
        if retained_items <= MAX_STORED_ITEMS:
            return

        terminal_jobs = sorted(
            (record for record in self._jobs.values() if self._is_terminal(record)),
            key=lambda record: (
                record.finished_at or record.updated_at,
                record.created_at,
                record.job_id,
            ),
        )
        for record in terminal_jobs:
            if retained_items <= MAX_STORED_ITEMS:
                break
            del self._jobs[record.job_id]
            retained_items -= record.total_items

    def _group_batch(self, batch: list[_QueuedItem]) -> dict[str, list[_QueuedItem]]:
        grouped: dict[str, list[_QueuedItem]] = defaultdict(list)
        for item in batch:
            grouped[item.options_signature].append(item)
        return grouped

    def _prepare_signature_batch(
        self, signature_items: Sequence[_QueuedItem]
    ) -> tuple[
        PreparedPredictionOptions | None,
        list[tuple[PredictJobRecord, PredictJobItemResult]],
        list[Image.Image],
    ]:
        items = list(signature_items)
        if not items:
            return None, [], []

        options = items[0].options
        staged_records: list[tuple[PredictJobRecord, PredictJobItemResult]] = []
        images: list[Image.Image] = []
        for queued in items:
            record = self._jobs.get(queued.job_id)
            if record is None or record.status == JOB_STATUS_CANCELED:
                continue
            item_record = next(
                (entry for entry in record.items if entry.item_id == queued.item_id),
                None,
            )
            if item_record is None:
                continue
            self._mark_item_running(item_record, record)
            try:
                images.append(self._decode_image(queued.payload))
            except BadRequestError as exc:
                self._mark_item_failed(item_record, record, str(exc))
                continue
            staged_records.append((record, item_record))
        return options, staged_records, images

    def _apply_predictions(
        self,
        staged_records: list[tuple[PredictJobRecord, PredictJobItemResult]],
        predictions: Sequence[Sequence[object]],
    ) -> None:
        for (record, item_record), item_predictions in zip(staged_records, predictions):
            normalized_predictions: list[tuple[str, float]] = []
            for prediction in item_predictions:
                canonical_tag = getattr(prediction, "canonical_tag", None)
                score = getattr(prediction, "score", None)
                if canonical_tag is None or score is None:
                    continue
                normalized_predictions.append((canonical_tag, float(score)))
            self._mark_item_done(item_record, record, normalized_predictions)

    def _process_signature_batch(self, signature_items: Sequence[_QueuedItem]) -> None:
        options, staged_records, images = self._prepare_signature_batch(signature_items)
        if options is None or not images:
            return
        predictions = self._context.prediction_service.score_images(
            images=tuple(images),
            canonical_tags=options.canonical_tags,
            extra_labels=options.extra_tags,
            min_score=options.min_score,
            limit=options.limit,
        )
        self._apply_predictions(staged_records, predictions)

    def _process_batch(self, batch: list[_QueuedItem]) -> None:
        if not batch:
            return
        grouped = self._group_batch(batch)
        for signature_items in grouped.values():
            self._process_signature_batch(signature_items)

    def _take_pending_batch(
        self, pending: dict[str, list[_QueuedItem]]
    ) -> list[_QueuedItem] | None:
        if not pending:
            return None
        signature = min(
            pending.keys(),
            key=lambda key: self._jobs[pending[key][0].job_id].created_at,
        )
        batch = pending[signature][: self._batch_size]
        del pending[signature][: self._batch_size]
        if not pending[signature]:
            del pending[signature]
        return batch

    def _drain_pending_queue(self, pending: dict[str, list[_QueuedItem]]) -> None:
        while True:
            try:
                drained = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if drained is None:
                self._stopping = True
                return
            pending[drained.options_signature].append(drained)

    async def _worker(self) -> None:
        pending: dict[str, list[_QueuedItem]] = defaultdict(list)
        while not self._stopping:
            queued = await self._queue.get()
            if queued is None:
                break
            pending[queued.options_signature].append(queued)

            if self._flush_interval_seconds:
                await asyncio.sleep(self._flush_interval_seconds)

            self._drain_pending_queue(pending)

            while pending and not self._stopping:
                batch = self._take_pending_batch(pending)
                if batch is None:
                    break
                self._process_batch(batch)

    async def _reaper(self) -> None:
        while not self._stopping:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            self._cleanup_retained_jobs(trim_to_capacity=True)
