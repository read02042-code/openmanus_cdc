import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.agent.control_measures import ControlMeasuresAgent
from app.agent.plan_validation import PlanValidationAgent
from app.agent.resource_allocation import ResourceAllocationAgent
from app.agent.risk_assessment import RiskAssessmentAgent
from app.config import config
from app.flow.cdc_plan_flow import CDCPlanFlow
from app.tool.cdc_plan_export import CDCPlanExportTool


class CDCPlanRunRequest(BaseModel):
    disease_type: str = Field(default="covid19")
    location: str = Field(default="某中学")
    population: int = Field(default=3000, ge=1)
    reported_cases: int = Field(default=25, ge=0)
    underreport_factor: float = Field(default=1.5, gt=0)
    days: int = Field(default=7, ge=1, le=30)
    jurisdiction: str = Field(default="某市疾控中心")
    region_profile: Optional[str] = Field(
        default=None,
        description="Optional local context for supplementary (regional) measures, e.g. medical capacity, population structure, events, policies.",
    )
    output_format: str = Field(default="docx", description="docx|pdf")
    output_path: Optional[str] = Field(default=None)
    output_docx: Optional[str] = Field(
        default=None, description="Backward compatible alias of output_path"
    )


class CDCPlanRunResponse(BaseModel):
    plan_id: str
    status: str


class CDCPlanStatusResponse(BaseModel):
    plan_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None
    plan_text: Optional[str] = None


class ExportRequest(BaseModel):
    output_format: str = Field(default="docx", description="docx|pdf")
    output_path: Optional[str] = Field(default=None)
    output_docx: Optional[str] = Field(
        default=None, description="Backward compatible alias of output_path"
    )


class ExportResponse(BaseModel):
    plan_id: str
    output_path: str
    note: Optional[str] = None


class ManualFixRequest(BaseModel):
    action: str = Field(description="e/s/x/q")
    plan: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional edited plan JSON"
    )


class ManualFixStatusResponse(BaseModel):
    plan_id: str


class ReviewRequest(BaseModel):
    action: str = Field(description="approve/edit/cancel")
    plan: Optional[Dict[str, Any]] = Field(default=None)


class ReviewStatusResponse(BaseModel):
    plan_id: str
    pending: bool
    draft_path: Optional[str] = None


@dataclass
class _RunRecord:
    plan_id: str
    flow: CDCPlanFlow
    task: Optional["asyncio.Task[None]"]
    status: str = "running"
    error: Optional[str] = None
    output_path: Optional[str] = None


app = FastAPI(title="CDC Plan API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RUNS: Dict[str, _RunRecord] = {}
_RUNS_LOCK = asyncio.Lock()


def _build_risk_prompt(req: CDCPlanRunRequest) -> str:
    return (
        f"disease_type: {req.disease_type}；location: {req.location}；population: {req.population}；"
        f"reported_cases: {req.reported_cases}；underreport_factor: {req.underreport_factor}；days: {req.days}；"
        "请输出风险等级与预测，并说明E/I/R如何确定。"
    )


def _new_plan_id() -> str:
    ts = int(time.time() * 1000)
    return f"cdc_web_{ts}_{uuid4().hex[:6]}"


def _normalize_output_path(
    plan_id: str, output_path: Optional[str], output_format: str
) -> str:
    fmt = str(output_format or "").strip().lower()
    if fmt not in {"docx", "pdf"}:
        fmt = "docx"
    if output_path and str(output_path).strip():
        s = str(output_path).strip()
        lower = s.lower()
        if lower.endswith(".docx") or lower.endswith(".pdf"):
            return s
        return f"{s}.{fmt}"
    return f"{plan_id}.{fmt}"


def _extract_output_path(export_output: Any) -> Optional[str]:
    if export_output is None:
        return None
    if isinstance(export_output, dict):
        return str(export_output.get("output_path") or "") or None
    if isinstance(export_output, str):
        s = export_output.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and obj.get("output_path"):
                return str(obj["output_path"])
        except Exception:
            return None
    return None


async def _plan_text(flow: CDCPlanFlow) -> str:
    res = await flow.planning_tool.execute(command="get", plan_id=flow.active_plan_id)
    return res.output or ""


async def _run_flow(record: _RunRecord, prompt: str) -> None:
    try:
        await record.flow.execute(prompt)
        record.status = "done"
        record.output_path = _extract_output_path(record.flow.ctx.get("export"))
    except Exception as e:
        record.status = "error"
        record.error = str(e)


@app.get("/")
async def index() -> JSONResponse:
    return JSONResponse(
        content={
            "service": "cdc-plan-api",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@app.post("/cdc/plan/run", response_model=CDCPlanRunResponse)
async def run_plan(req: CDCPlanRunRequest) -> CDCPlanRunResponse:
    plan_id = _new_plan_id()
    out_path_raw = req.output_path or req.output_docx
    output_path = _normalize_output_path(plan_id, out_path_raw, req.output_format)
    prompt = _build_risk_prompt(req)

    flow = CDCPlanFlow(
        agents={
            "risk": RiskAssessmentAgent(),
            "measures": ControlMeasuresAgent(),
            "resources": ResourceAllocationAgent(),
            "validation": PlanValidationAgent(),
        },
        active_plan_id=plan_id,
        output_docx=output_path,
    )
    flow.ctx["review_mode"] = "api"
    if req.region_profile and str(req.region_profile).strip():
        flow.ctx["region_profile"] = str(req.region_profile).strip()

    record = _RunRecord(plan_id=plan_id, flow=flow, task=None)  # type: ignore[arg-type]
    task = asyncio.create_task(_run_flow(record, prompt))
    record.task = task

    async with _RUNS_LOCK:
        _RUNS[plan_id] = record

    task.add_done_callback(lambda _: None)
    return CDCPlanRunResponse(plan_id=plan_id, status="running")


@app.get("/cdc/plan/{plan_id}", response_model=CDCPlanStatusResponse)
async def plan_status(plan_id: str) -> CDCPlanStatusResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    text = await _plan_text(record.flow)
    status = record.status
    if status == "running" and record.flow.ctx.get("review_pending"):
        status = "waiting_review"
    return CDCPlanStatusResponse(
        plan_id=plan_id,
        status=status,
        error=record.error,
        output_path=record.output_path,
        plan_text=text,
    )


@app.get("/cdc/plan/{plan_id}/plan.json")
async def get_plan_json(plan_id: str) -> JSONResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    plan_obj = record.flow.ctx.get("plan")
    if not isinstance(plan_obj, dict):
        raise HTTPException(status_code=400, detail="plan not ready")
    return JSONResponse(content=plan_obj)


@app.get("/cdc/plan/{plan_id}/draft.json")
async def get_draft_json(plan_id: str) -> JSONResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    draft_obj = record.flow.ctx.get("draft_plan") or record.flow.ctx.get("plan")
    if not isinstance(draft_obj, dict):
        raise HTTPException(status_code=400, detail="draft not ready")
    return JSONResponse(content=draft_obj)


@app.post("/cdc/plan/{plan_id}/export", response_model=ExportResponse)
async def export_plan(plan_id: str, req: ExportRequest) -> ExportResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)

    plan_obj: Optional[Dict[str, Any]] = None
    if record is not None:
        draft_obj = record.flow.ctx.get("draft_plan") or record.flow.ctx.get("plan")
        if isinstance(draft_obj, dict):
            plan_obj = draft_obj

    if plan_obj is None:
        p = config.workspace_root / f"{plan_id}_draft_plan.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="draft file not found")
        try:
            plan_obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid draft json: {e}")
        if not isinstance(plan_obj, dict):
            raise HTTPException(status_code=400, detail="draft json is not an object")

    out_path_raw = req.output_path or req.output_docx
    output_path = _normalize_output_path(plan_id, out_path_raw, req.output_format)
    tool = CDCPlanExportTool()
    res = await tool.execute(
        plan=plan_obj, output_path=output_path, output_format=req.output_format
    )
    if res.error:
        raise HTTPException(status_code=500, detail=res.error)

    out_file = output_path
    note: Optional[str] = None
    if isinstance(res.output, dict):
        out_file = str(res.output.get("output_path") or output_path)
        note = res.output.get("note")
    elif isinstance(res.output, str):
        s = res.output.strip()
        if s:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out_file = str(obj.get("output_path") or output_path)
                    note = obj.get("note")
            except Exception:
                pass

    if record is not None:
        record.output_path = str(out_file)
        record.error = None
        if record.status in {"error", "running"}:
            record.status = "done"

    return ExportResponse(plan_id=plan_id, output_path=str(out_file), note=note)


@app.get("/cdc/plan/{plan_id}/review", response_model=ReviewStatusResponse)
async def review_status(plan_id: str) -> ReviewStatusResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    pending = bool(record.flow.ctx.get("review_pending"))
    draft_path = record.flow.ctx.get("draft_path")
    return ReviewStatusResponse(
        plan_id=plan_id,
        pending=pending,
        draft_path=str(draft_path) if draft_path else None,
    )


@app.post("/cdc/plan/{plan_id}/review")
async def review_submit(plan_id: str, req: ReviewRequest) -> JSONResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    flow = record.flow
    event = flow.ctx.get("review_event")
    if not isinstance(event, asyncio.Event):
        raise HTTPException(status_code=400, detail="review is not pending")
    action = str(req.action or "").strip().lower()
    if action not in {"approve", "edit", "cancel"}:
        raise HTTPException(status_code=400, detail="invalid action")
    flow.ctx["review_action"] = action
    if req.plan is not None:
        flow.ctx["review_plan_override"] = req.plan
    event.set()
    return JSONResponse(content={"ok": True})


@app.get("/cdc/plan/{plan_id}/manual_fix", response_model=ManualFixStatusResponse)
async def manual_fix_status(plan_id: str) -> ManualFixStatusResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    pending = bool(record.flow.ctx.get("manual_fix_pending"))
    path = record.flow.ctx.get("manual_fix_path")
    return ManualFixStatusResponse(
        plan_id=plan_id,
        pending=pending,
        manual_fix_path=str(path) if path else None,
    )


@app.post("/cdc/plan/{plan_id}/manual_fix")
async def manual_fix_submit(plan_id: str, req: ManualFixRequest) -> JSONResponse:
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="plan_id not found")
    flow = record.flow
    event = flow.ctx.get("manual_fix_event")
    if not isinstance(event, asyncio.Event):
        raise HTTPException(status_code=400, detail="manual fix is not pending")

    action = str(req.action or "").strip().lower()
    if action not in {"e", "s", "x", "q"}:
        raise HTTPException(status_code=400, detail="invalid action")
    flow.ctx["manual_fix_action"] = action
    if req.plan is not None:
        flow.ctx["manual_fix_plan_override"] = req.plan
    event.set()
    return JSONResponse(content={"ok": True})


@app.get("/cdc/plan/{plan_id}/download")
async def download(plan_id: str):
    async with _RUNS_LOCK:
        record = _RUNS.get(plan_id)
    if record is None:
        ws = Path(config.workspace_root)
        candidates = list(ws.glob(f"{plan_id}*.pdf")) + list(
            ws.glob(f"{plan_id}*.docx")
        )
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            raise HTTPException(status_code=404, detail="plan_id not found")
        p = max(candidates, key=lambda x: x.stat().st_mtime)
    else:
        output_path = record.output_path
        if not output_path:
            raise HTTPException(status_code=400, detail="no output_path yet")
        p = Path(output_path)
        if not p.is_absolute():
            p = config.workspace_root / p
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"file not found: {p}")
    filename = p.name
    lower = filename.lower()
    media_type = "application/octet-stream"
    if lower.endswith(".docx"):
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    elif lower.endswith(".pdf"):
        media_type = "application/pdf"
    return FileResponse(
        path=str(p),
        media_type=media_type,
        filename=filename,
    )


@app.get("/cdc/runs")
async def list_runs():
    async with _RUNS_LOCK:
        items = [
            {
                "plan_id": r.plan_id,
                "status": r.status,
                "error": r.error,
                "output_path": r.output_path,
            }
            for r in _RUNS.values()
        ]
    return JSONResponse(content={"runs": items})
