from __future__ import annotations

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, update, func
from sqlalchemy.orm import load_only
from app.storage.db import async_session
from app.storage.models import Job

debug_router_jobs = APIRouter(prefix="/debug", tags=["debug"])

@debug_router_jobs.get("/jobs")
async def debug_jobs_list(
    status: str = Query(default="queued", description="queued|running|done|error"),
    limit: int = Query(default=50, ge=1, le=500),
):
    async with async_session() as session:
        stmt = (
            select(Job)
            .options(load_only(Job.id, Job.job_type, Job.status, Job.run_after, Job.attempts, Job.max_attempts, Job.last_error))
            .where(Job.status == status)
            .order_by(Job.id.desc())
            .limit(limit)
        )
        res = await session.execute(stmt)
        items = res.scalars().all()

    return {
        "ok": True,
        "count": len(items),
        "items": [
            {
                "id": j.id,
                "job_type": j.job_type,
                "status": j.status,
                "run_after": j.run_after,
                "attempts": j.attempts,
                "max_attempts": j.max_attempts,
                "last_error": j.last_error,
            }
            for j in items
        ],
    }


@debug_router_jobs.get("/jobs/{job_id}")
async def debug_jobs_get(job_id: int):
    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "ok": True,
            "job": {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "run_after": job.run_after,
                "attempts": job.attempts,
                "max_attempts": job.max_attempts,
                "last_error": job.last_error,
                "payload": job.payload,
                "locked_at": job.locked_at,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            },
        }


@debug_router_jobs.post("/jobs/{job_id}/requeue")
async def debug_jobs_requeue(job_id: int, seconds: int = Query(default=0, ge=0, le=3600)):
    """
    Принудительно возвращает job в queued (например если застряла в error/running).
    """
    run_after = datetime.utcnow() + timedelta(seconds=seconds)

    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status="queued", run_after=run_after, locked_at=None, updated_at=func.now())
        )
        await session.commit()

    return {"ok": True, "job_id": job_id, "status": "queued", "run_after": run_after}

