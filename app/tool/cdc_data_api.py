from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import PrivateAttr

from app.cdc.cases_store import CaseReport, CaseStore
from app.cdc.materials_store import MaterialStore
from app.config import config
from app.tool.base import BaseTool, ToolResult


class CDCDataAPITool(BaseTool):
    name: str = "cdc_data_api"
    description: str = (
        "Query simulated CDC internal databases for anonymized case stats and material stock."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": [
                    "cases_query",
                    "cases_summary",
                    "cases_append",
                    "materials_list",
                    "materials_get_stock",
                    "materials_allocate",
                    "materials_upsert",
                    "reset_demo_data",
                ],
                "description": "Operation to execute",
            },
            "event_type": {"type": "string", "description": "Event type filter"},
            "location_contains": {
                "type": "string",
                "description": "Location substring filter",
            },
            "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
            "limit": {
                "type": "integer",
                "description": "Max rows for cases_query",
                "default": 200,
            },
            "report_date": {
                "type": "string",
                "description": "Report date (YYYY-MM-DD) for cases_append",
            },
            "location": {"type": "string", "description": "Location for cases_append"},
            "confirmed_cases": {
                "type": "integer",
                "description": "Confirmed cases for cases_append",
            },
            "suspected_cases": {
                "type": "integer",
                "description": "Suspected cases for cases_append",
            },
            "severe_cases": {
                "type": "integer",
                "description": "Severe cases for cases_append",
            },
            "deaths": {"type": "integer", "description": "Deaths for cases_append"},
            "notes": {"type": "string", "description": "Notes for cases_append"},
            "warehouse_id": {
                "type": "string",
                "description": "Warehouse filter or target",
            },
            "sku": {"type": "string", "description": "Material SKU"},
            "name": {
                "type": "string",
                "description": "Material name (exact or substring)",
            },
            "quantity": {
                "type": "number",
                "description": "Quantity for allocate or set stock",
            },
            "unit": {"type": "string", "description": "Unit for upsert"},
            "category": {"type": "string", "description": "Category for upsert"},
            "spec": {"type": "string", "description": "Spec for upsert"},
            "safety_stock": {
                "type": "number",
                "description": "Safety stock for upsert",
            },
            "materials_path": {
                "type": "string",
                "description": "Materials DB path (absolute or relative to workspace)",
            },
            "cases_path": {
                "type": "string",
                "description": "Cases DB path (absolute or relative to workspace)",
            },
            "persist": {
                "type": "boolean",
                "description": "Whether to persist changes back to storage",
                "default": True,
            },
        },
        "required": ["command"],
    }

    _materials: Optional[MaterialStore] = PrivateAttr(default=None)
    _cases: Optional[CaseStore] = PrivateAttr(default=None)
    _materials_path: Optional[Path] = PrivateAttr(default=None)
    _cases_path: Optional[Path] = PrivateAttr(default=None)

    @staticmethod
    def _as_path(path_str: Optional[str], default_path: Path) -> Path:
        if not path_str:
            return default_path
        p = Path(path_str)
        if p.is_absolute():
            return p
        return config.workspace_root / p

    def _load(self, *, materials_path: Path, cases_path: Path) -> None:
        if self._materials is None or self._materials_path != materials_path:
            self._materials = MaterialStore.load_or_create(materials_path)
            self._materials_path = materials_path
        if self._cases is None or self._cases_path != cases_path:
            self._cases = CaseStore.load_or_create(cases_path)
            self._cases_path = cases_path

    def _persist_if_needed(self, *, persist: bool) -> None:
        if not persist:
            return
        if self._materials and self._materials_path:
            self._materials.save(self._materials_path)
        if self._cases and self._cases_path:
            self._cases.save(self._cases_path)

    async def execute(self, **kwargs) -> ToolResult:
        command = str(kwargs.get("command") or "").strip()
        persist = bool(kwargs.get("persist", True))

        materials_path = self._as_path(
            kwargs.get("materials_path"), MaterialStore.default_path()
        )
        cases_path = self._as_path(kwargs.get("cases_path"), CaseStore.default_path())

        self._load(materials_path=materials_path, cases_path=cases_path)
        materials = self._materials
        cases = self._cases
        if materials is None or cases is None:
            return ToolResult(error="failed to initialize data stores")

        if command == "reset_demo_data":
            self._materials = MaterialStore.default()
            self._cases = CaseStore.default()
            self._materials_path = materials_path
            self._cases_path = cases_path
            self._persist_if_needed(persist=persist)
            return self.success_response(
                {
                    "status": "ok",
                    "materials_path": str(materials_path),
                    "cases_path": str(cases_path),
                }
            )

        if command == "cases_query":
            rows = cases.query(
                event_type=kwargs.get("event_type"),
                location_contains=kwargs.get("location_contains"),
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
                limit=int(kwargs.get("limit", 200) or 200),
            )
            return self.success_response(
                {
                    "count": len(rows),
                    "rows": [r.model_dump() for r in rows],
                }
            )

        if command == "cases_summary":
            summary = cases.summarize(
                event_type=kwargs.get("event_type"),
                location_contains=kwargs.get("location_contains"),
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
            )
            return self.success_response(summary)

        if command == "cases_append":
            report_date = kwargs.get("report_date")
            event_type = kwargs.get("event_type")
            location = kwargs.get("location")
            if not report_date or not event_type or not location:
                return ToolResult(
                    error="report_date, event_type and location are required for cases_append"
                )
            report = CaseReport(
                report_date=str(report_date),
                event_type=str(event_type),
                location=str(location),
                confirmed_cases=int(kwargs.get("confirmed_cases") or 0),
                suspected_cases=int(kwargs.get("suspected_cases") or 0),
                severe_cases=int(kwargs.get("severe_cases") or 0),
                deaths=int(kwargs.get("deaths") or 0),
                notes=kwargs.get("notes"),
            )
            cases.append_report(report)
            self._persist_if_needed(persist=persist)
            return self.success_response(
                {
                    "status": "ok",
                    "count": len(cases.reports),
                    "report": report.model_dump(),
                }
            )

        if command == "materials_list":
            stock = materials.list_stock(warehouse_id=kwargs.get("warehouse_id"))
            return self.success_response(
                {
                    "warehouses": [w.model_dump() for w in materials.warehouses],
                    "items": [i.model_dump() for i in materials.items],
                    "stock": stock,
                }
            )

        if command == "materials_get_stock":
            item = materials.find_item(sku=kwargs.get("sku"), name=kwargs.get("name"))
            if not item:
                return ToolResult(error="material not found by sku/name")
            total = materials.get_total_stock(sku=item.sku)
            stock = materials.list_stock(warehouse_id=kwargs.get("warehouse_id"))
            stock = [s for s in stock if s["sku"] == item.sku]
            return self.success_response(
                {"item": item.model_dump(), "total_quantity": total, "stock": stock}
            )

        if command == "materials_upsert":
            sku = kwargs.get("sku")
            name = kwargs.get("name")
            if not sku or not name:
                return ToolResult(error="sku and name are required")
            item = materials.upsert_item(
                sku=str(sku),
                name=str(name),
                unit=str(kwargs.get("unit") or "unit"),
                category=kwargs.get("category"),
                spec=kwargs.get("spec"),
                safety_stock=kwargs.get("safety_stock"),
            )
            warehouse_id = kwargs.get("warehouse_id")
            if warehouse_id and kwargs.get("quantity") is not None:
                materials.set_stock(
                    warehouse_id=str(warehouse_id),
                    sku=item.sku,
                    quantity=float(kwargs.get("quantity") or 0),
                )
            self._persist_if_needed(persist=persist)
            return self.success_response(
                {
                    "item": item.model_dump(),
                    "materials_path": str(materials_path),
                }
            )

        if command == "materials_allocate":
            item = materials.find_item(sku=kwargs.get("sku"), name=kwargs.get("name"))
            if not item:
                return ToolResult(error="material not found by sku/name")
            qty = kwargs.get("quantity")
            if qty is None:
                return ToolResult(error="quantity is required")
            allocations, allocated = materials.allocate(
                sku=item.sku,
                quantity=float(qty),
                warehouse_id=kwargs.get("warehouse_id"),
            )
            self._persist_if_needed(persist=persist)
            return self.success_response(
                {
                    "sku": item.sku,
                    "name": item.name,
                    "requested_quantity": float(qty),
                    "allocated_quantity": float(allocated),
                    "allocations": allocations,
                    "materials_path": str(materials_path),
                }
            )

        return ToolResult(error=f"unsupported command: {command}")
