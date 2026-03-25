"""In-memory prediction job queue and micro-batching."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import TYPE_CHECKING, Sequence
from uuid import uuid4

from PIL import Image
from fastapi import UploadFile

from ..errors import BadRequestError, NotFoundError
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
        self._stopping = False

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._stopping = False
        self._worker_task = asyncio.create_task(
            self._worker(), name="predict-job-worker"
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._worker_task is None:
            return
        await self._queue.put(None)
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        finally:
            self._worker_task = None

    async def submit_job(
        self,
        *,
        files: Sequence[UploadFile],
        options: PreparedPredictionOptions,
    ) -> PredictJobRecord:
        if not files:
            raise BadRequestError("at least one file is required")

        payloads: list[tuple[str, bytes]] = []
        for upload in files:
            filename = (upload.filename or "").strip()
            if not filename:
                raise BadRequestError("each uploaded file must have a filename")
            payload = await upload.read()
            if not payload:
                raise BadRequestError(f"file '{filename}' is empty")
            payloads.append((filename, payload))

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
        record = self._jobs.get(job_id)
        if record is None:
            raise NotFoundError(f"prediction job '{job_id}' not found")
        return self._snapshot(record)

    def cancel_job(self, job_id: str) -> PredictJobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise NotFoundError(f"prediction job '{job_id}' not found")
        if record.status not in {
            JOB_STATUS_DONE,
            JOB_STATUS_PARTIAL,
            JOB_STATUS_FAILED,
        }:
            record.status = JOB_STATUS_CANCELED
            self._touch(record)
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
    def _snapshot(record: PredictJobRecord) -> PredictJobRecord:
        return PredictJobRecord(
            job_id=record.job_id,
            status=record.status,
            total_items=record.total_items,
            completed_items=record.completed_items,
            failed_items=record.failed_items,
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
            return (
                JOB_STATUS_FAILED
                if record.failed_items == record.total_items
                else (
                    JOB_STATUS_PARTIAL if record.failed_items > 0 else JOB_STATUS_DONE
                )
            )
        if record.completed_items > 0 or record.failed_items > 0:
            return JOB_STATUS_RUNNING
        return JOB_STATUS_QUEUED

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
        self._touch(record)

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

    async def _process_batch(self, batch: list[_QueuedItem]) -> None:
        if not batch:
            return
        grouped: dict[str, list[_QueuedItem]] = defaultdict(list)
        for item in batch:
            grouped[item.options_signature].append(item)

        for signature_items in grouped.values():
            items = list(signature_items)
            options = items[0].options
            records = []
            images: list[Image.Image] = []
            for queued in items:
                record = self._jobs.get(queued.job_id)
                if record is None or record.status == JOB_STATUS_CANCELED:
                    continue
                item_record = next(
                    (
                        entry
                        for entry in record.items
                        if entry.item_id == queued.item_id
                    ),
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
                records.append((record, item_record))

            if not images:
                continue

            predictions = self._context.prediction_service.score_images(
                images=tuple(images),
                canonical_tags=options.canonical_tags,
                extra_labels=options.extra_tags,
                min_score=options.min_score,
                limit=options.limit,
            )

            for (record, item_record), item_predictions in zip(records, predictions):
                self._mark_item_done(
                    item_record,
                    record,
                    [(pred.canonical_tag, pred.score) for pred in item_predictions],
                )

    async def _worker(self) -> None:
        pending: dict[str, list[_QueuedItem]] = defaultdict(list)
        try:
            while not self._stopping:
                queued = await self._queue.get()
                if queued is None:
                    break
                pending[queued.options_signature].append(queued)

                if self._flush_interval_seconds:
                    await asyncio.sleep(self._flush_interval_seconds)

                while True:
                    try:
                        drained = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if drained is None:
                        self._stopping = True
                        break
                    pending[drained.options_signature].append(drained)

                while pending:
                    signature = min(
                        pending.keys(),
                        key=lambda key: self._jobs[pending[key][0].job_id].created_at,
                    )
                    batch = pending[signature][: self._batch_size]
                    del pending[signature][: self._batch_size]
                    if not pending[signature]:
                        del pending[signature]
                    await self._process_batch(batch)
                    if self._stopping:
                        break
        except asyncio.CancelledError:
            raise
