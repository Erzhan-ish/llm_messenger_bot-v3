from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update, func
from sqlalchemy.orm import load_only

from app.storage.db import AsyncSessionLocal
from app.storage.models import Job
from app.logging import logger


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def enqueue_job(
    job_type: str,
    payload: dict[str, Any],
    run_after: datetime | None = None,
    max_attempts: int = 5,
) -> int:
    run_after = run_after or utcnow()

    async with AsyncSessionLocal() as session:
        job = Job(
            job_type=job_type,
            payload=payload,
            status="queued",
            run_after=run_after,
            max_attempts=max_attempts,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        logger.info("Job enqueued | id={} | type={}", job.id, job_type)
        return job.id


async def fetch_and_lock_jobs(
    limit: int = 10,
    job_types: list[str] | None = None,
) -> list[Job]:
    """
    Забираем queued jobs, готовые к запуску, и лочим их:
    SELECT ... FOR UPDATE SKIP LOCKED + переводим в running.

    job_types: если задан, фильтруем по типу (напр. ["inbound", "stt"]).
    """
    async with AsyncSessionLocal() as session:
        conditions = [
            Job.status == "queued",
            Job.run_after <= func.now(),
        ]
        if job_types is not None:
            conditions.append(Job.job_type.in_(job_types))

        stmt = (
            select(Job)
            .options(load_only(Job.id, Job.job_type, Job.payload, Job.attempts, Job.max_attempts))
            .where(*conditions)
            .order_by(Job.id.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )

        res = await session.execute(stmt)
        jobs = list(res.scalars().all())

        if not jobs:
            return []

        ids = [j.id for j in jobs]

        await session.execute(
            update(Job)
            .where(Job.id.in_(ids))
            .values(
                status="running",
                locked_at=func.now(),
                updated_at=func.now(),
            )
        )
        await session.commit()

        return jobs


async def mark_done(job_id: int) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status="done", updated_at=func.now())
        )
        await session.commit()


async def mark_error(job_id: int, error_text: str) -> None:
    """
    Финальная ошибка (без retry).
    attempts тут НЕ увеличиваем, чтобы не ломать retry_or_give_up().
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="error",
                last_error=error_text[:2000],
                updated_at=func.now(),
            )
        )
        await session.commit()


async def has_newer_active_job(
    current_job_id: int,
    channel: str,
    external_user_id: str,
) -> bool:
    """True if there is a newer queued OR running inbound job for the same channel+user.

    Checks both 'queued' and 'running' so that burst messages picked up in the
    same worker batch are still properly suppressed: when multiple jobs for the
    same user are marked 'running' simultaneously, only the highest-id job
    (the latest message) should produce a reply.

    Uses channel + external_user_id to avoid cross-channel collisions
    (e.g. Telegram user 123 vs WhatsApp user 123).
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            select(func.count())
            .select_from(Job)
            .where(
                Job.status.in_(["queued", "running"]),
                Job.job_type == "inbound",
                Job.id > current_job_id,
                Job.payload["channel"].as_string() == channel,
                Job.payload["external_user_id"].as_string() == external_user_id,
            )
        )
        res = await session.execute(stmt)
        return (res.scalar() or 0) > 0


async def has_newer_queued_job(
    current_job_id: int,
    channel: str,
    external_user_id: str,
) -> bool:
    """True if there is a newer queued inbound job for the same channel+user."""
    return await has_newer_active_job(current_job_id, channel, external_user_id)


async def defer_job(
    job_id: int,
    *,
    delay_seconds: int = 2,
    reason: str = "conversation_locked",
) -> None:
    """Re-queue a running job to retry after delay_seconds without counting it as an attempt."""
    run_after = utcnow() + timedelta(seconds=delay_seconds)

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="queued",
                run_after=run_after,
                locked_at=None,
                last_error=reason,
                updated_at=func.now(),
            )
        )
        await session.commit()


async def requeue_stale_running_jobs(*, stale_after_seconds: int) -> int:
    """Find running jobs whose locked_at is older than stale_after_seconds and requeue them."""
    now = utcnow()
    stale_cutoff = now - timedelta(seconds=stale_after_seconds)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            update(Job)
            .where(
                Job.status == "running",
                Job.locked_at < stale_cutoff,
            )
            .values(
                status="queued",
                locked_at=None,
                run_after=func.now(),
                updated_at=func.now(),
            )
        )
        await session.commit()
        requeued = result.rowcount or 0

    if requeued:
        from app.logging import logger
        logger.info("StaleJobsRecovery | requeued={}", requeued)
    return requeued


async def retry_or_give_up(
    job_id: int,
    error_text: str,
    backoff_seconds: int = 10,
) -> None:
    """
    attempts+1 < max_attempts => queued + run_after+backoff
    иначе => error финально
    """
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            return

        next_attempt = int(job.attempts or 0) + 1

        if next_attempt >= int(job.max_attempts or 0):
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="error",
                    attempts=next_attempt,
                    last_error=error_text[:2000],
                    updated_at=func.now(),
                )
            )
            await session.commit()
            return

        run_after = utcnow() + timedelta(seconds=backoff_seconds)

        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="queued",
                attempts=next_attempt,
                last_error=error_text[:2000],
                run_after=run_after,
                updated_at=func.now(),
            )
        )
        await session.commit()
