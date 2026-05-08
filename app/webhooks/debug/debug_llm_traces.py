from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from app.config import settings

debug_router_llm_traces = APIRouter(prefix="/debug", tags=["debug"])


def _guard():
    if not settings.ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")


@debug_router_llm_traces.get("/llm/traces")
async def get_llm_traces(limit: int = Query(default=20, ge=1, le=200)):
    _guard()
    from app.services.llm_trace import get_tracer
    tracer = get_tracer()
    records = tracer.read_recent(limit=limit)
    return {"ok": True, "count": len(records), "traces": records}


@debug_router_llm_traces.get("/llm/traces/{trace_id}")
async def get_llm_trace(trace_id: str):
    _guard()
    from app.services.llm_trace import get_tracer
    tracer = get_tracer()
    record = tracer.get_by_id(trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"ok": True, "trace": record}
