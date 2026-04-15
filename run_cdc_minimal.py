import argparse
import asyncio
import json
import re
from datetime import date
from typing import Any, Dict, Optional

from app.agent.control_measures import ControlMeasuresAgent
from app.agent.plan_validation import PlanValidationAgent
from app.agent.resource_allocation import ResourceAllocationAgent
from app.agent.risk_assessment import RiskAssessmentAgent
from app.logger import logger
from app.tool.cdc_data_api import CDCDataAPITool
from app.tool.cdc_plan_export import CDCPlanExportTool


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_step_json(run_text: str, step_no: int = 1) -> Dict[str, Any]:
    if not isinstance(run_text, str) or not run_text.strip():
        return {}
    pattern = rf"Step {step_no}:\s*(\{{[\s\S]*?\}})(?:\nStep \d+:|\Z)"
    m = re.search(pattern, run_text)
    if not m:
        tail = run_text.split(f"Step {step_no}:", 1)
        if len(tail) == 2:
            return _safe_json_loads(tail[1].strip())
        return {}
    return _safe_json_loads(m.group(1).strip())


def _require_nonempty_list(v: Any, label: str) -> list:
    if isinstance(v, list) and len(v) > 0:
        return v
    raise RuntimeError(f"{label} is missing or empty")


def _print_thinking(label: str, items: list) -> None:
    print(f"\n==== {label} ====")
    print(json.dumps(items, ensure_ascii=False, indent=2))


def _cn_disease_name(disease_type: str) -> str:
    v = (disease_type or "").strip().lower()
    if v == "covid19":
        return "新冠"
    if v == "influenza":
        return "流感"
    if v == "norovirus":
        return "诺如"
    return "传染病"


def _build_plan_skeleton(
    *,
    disease_type: str,
    location: str,
    population: int,
    reported_cases: int,
    r0: Optional[float],
    incubation_days: Optional[float],
    infectious_days: Optional[float],
    risk_level: str,
    risk_summary: str,
    predicted_cases_7d: Optional[int],
    measures: list,
    resources_items: list,
    jurisdiction: Optional[str],
) -> Dict[str, Any]:
    created_at = date.today().isoformat()
    cn_name = _cn_disease_name(disease_type)
    title = f"{location}{cn_name}疫情应急处置预案"
    plan: Dict[str, Any] = {
        "meta": {
            "title": title,
            "jurisdiction": jurisdiction or "某市疾控中心",
            "created_at": created_at,
        },
        "input": {
            "event_type": disease_type,
            "location": location,
            "population": population,
            "reported_cases": reported_cases,
            "report_date": created_at,
            "transmission": {
                "r0": r0,
                "incubation_days": incubation_days,
                "infectious_days": infectious_days,
            },
        },
        "risk": {
            "level": risk_level,
            "summary": risk_summary,
            "predicted_cases_7d": predicted_cases_7d,
        },
        "measures": measures,
        "resources": {"items": resources_items},
        "sections": [
            {
                "title": "一、总则",
                "paragraphs": [
                    "为科学、规范、有序开展疫情应急处置工作，保障公众健康和社会稳定，制定本预案。"
                ],
                "subsections": [],
            },
            {
                "title": "二、组织指挥体系",
                "paragraphs": [
                    "建立应急指挥体系，明确疾控、教育、医疗、社区等部门职责分工与联动机制。"
                ],
                "subsections": [],
            },
            {
                "title": "三、监测预警与信息报告",
                "paragraphs": [
                    "落实监测、预警、信息报告与通报机制，确保疫情早发现、早报告、早处置。"
                ],
                "subsections": [],
            },
            {
                "title": "四、应急响应与处置措施",
                "paragraphs": ["具体措施见“防控措施”部分（measures）。"],
                "subsections": [],
            },
            {
                "title": "五、资源保障与调配",
                "paragraphs": ["资源清单见“resources”，并建立快速调拨与补充机制。"],
                "subsections": [],
            },
            {
                "title": "六、后期处置与预案管理",
                "paragraphs": [
                    "疫情结束后开展评估总结，完善预案；定期组织培训与演练，提高应急能力。"
                ],
                "subsections": [],
            },
        ],
    }
    return plan


async def main():
    parser = argparse.ArgumentParser(
        description="Run CDC end-to-end plan generation: risk -> measures -> resources -> validation -> export"
    )
    parser.add_argument("--disease_type", type=str, required=False, default="covid19")
    parser.add_argument("--location", type=str, required=False, default="某中学")
    parser.add_argument("--population", type=int, required=False, default=3000)
    parser.add_argument("--reported_cases", type=int, required=False, default=25)
    parser.add_argument("--underreport_factor", type=float, required=False, default=1.5)
    parser.add_argument("--days", type=int, required=False, default=7)
    parser.add_argument(
        "--jurisdiction", type=str, required=False, default="某市疾控中心"
    )
    parser.add_argument(
        "--output_docx", type=str, required=False, default="cdc_plan_end_to_end.docx"
    )
    args = parser.parse_args()

    disease_type = str(args.disease_type).strip()
    location = str(args.location).strip()
    population = int(args.population)
    reported_cases = int(args.reported_cases)
    underreport_factor = float(args.underreport_factor)
    days = int(args.days)

    logger.warning("Processing end-to-end CDC plan generation...")

    data_tool = CDCDataAPITool()
    await data_tool.execute(command="reset_demo_data")

    risk_agent = RiskAssessmentAgent()
    risk_prompt = (
        f"disease_type: {disease_type}；location: {location}；population: {population}；"
        f"reported_cases: {reported_cases}；underreport_factor: {underreport_factor}；days: {days}；"
        "请输出风险等级与预测，并说明E/I/R如何确定。"
    )
    risk_text = await risk_agent.run(risk_prompt)
    risk_step = _extract_step_json(risk_text, 1)
    risk_level = (
        (
            (risk_step.get("assessment") or {}) if isinstance(risk_step, dict) else {}
        ).get("risk_level")
    ) or "medium"
    predicted_7d = (
        (
            (risk_step.get("assessment") or {}) if isinstance(risk_step, dict) else {}
        ).get("predicted_cases_7d")
    ) or None
    risk_summary = (
        (
            (risk_step.get("assessment") or {}) if isinstance(risk_step, dict) else {}
        ).get("summary")
    ) or "已完成风险评估。"
    transmission = (risk_step.get("input") or {}) if isinstance(risk_step, dict) else {}
    risk_thinking = _require_nonempty_list(
        (
            (risk_step.get("assessment") or {}) if isinstance(risk_step, dict) else {}
        ).get("thinking_summary"),
        "RiskAssessment.thinking_summary",
    )
    _print_thinking("RiskAssessment.thinking_summary", risk_thinking)

    measures_agent = ControlMeasuresAgent()
    measures_prompt = (
        f"disease_type: {disease_type}；location: {location}；risk_level: {risk_level}；"
        "请基于规范检索生成防控措施：至少5条，其中核心措施(core)不少于3条；"
        "每条core措施必须绑定>=1条引用(citations)；并输出thinking_summary。"
    )
    measures_text = await measures_agent.run(measures_prompt)
    measures_step = _extract_step_json(measures_text, 1)
    measures = (
        (
            (measures_step.get("output") or {})
            if isinstance(measures_step, dict)
            else {}
        ).get("measures")
    ) or []
    measures_thinking = _require_nonempty_list(
        (
            (measures_step.get("output") or {})
            if isinstance(measures_step, dict)
            else {}
        ).get("thinking_summary"),
        "ControlMeasures.thinking_summary",
    )
    _print_thinking("ControlMeasures.thinking_summary", measures_thinking)

    resources_agent = ResourceAllocationAgent()
    resources_prompt = (
        f"disease_type: {disease_type}；location: {location}；risk_level: {risk_level}；"
        f"population: {population}；cases: {reported_cases}；days: {days}；"
        "请给出物资需求清单（按7天），并尝试从库存中分配；输出shortages（若有）与thinking_summary。"
    )
    resources_text = await resources_agent.run(resources_prompt)
    resources_step = _extract_step_json(resources_text, 1)
    demands = resources_step.get("demands") if isinstance(resources_step, dict) else []
    resources_items = []
    if isinstance(demands, list):
        for d in demands:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "").strip()
            if not name:
                continue
            resources_items.append(
                {
                    "name": name,
                    "unit": "unit",
                    "quantity": float(d.get("quantity") or 0),
                }
            )
    resources_out = (
        (resources_step.get("output") or {}) if isinstance(resources_step, dict) else {}
    )
    alloc_thinking = _require_nonempty_list(
        (resources_out or {}).get("thinking_summary"),
        "ResourceAllocation.thinking_summary",
    )
    demands_thinking = _require_nonempty_list(
        (resources_out or {}).get("demands_thinking_summary"),
        "ResourceAllocation.demands_thinking_summary",
    )
    _print_thinking("ResourceAllocation.demands_thinking_summary", demands_thinking)
    _print_thinking("ResourceAllocation.thinking_summary", alloc_thinking)

    plan = _build_plan_skeleton(
        disease_type=disease_type,
        location=location,
        population=population,
        reported_cases=reported_cases,
        r0=transmission.get("r0"),
        incubation_days=transmission.get("incubation_days"),
        infectious_days=transmission.get("infectious_days"),
        risk_level=risk_level,
        risk_summary=risk_summary,
        predicted_cases_7d=predicted_7d,
        measures=measures,
        resources_items=resources_items,
        jurisdiction=str(args.jurisdiction).strip() if args.jurisdiction else None,
    )

    validation_agent = PlanValidationAgent()
    validation_text = await validation_agent.run(
        json.dumps({"plan": plan}, ensure_ascii=False)
    )
    validation_step = _extract_step_json(validation_text, 1)
    final_plan = plan
    if isinstance(validation_step, dict):
        output_obj = validation_step.get("output") or {}
        if isinstance(output_obj, dict) and isinstance(
            output_obj.get("improved_plan"), dict
        ):
            final_plan = output_obj["improved_plan"]
    validation_output = (
        (validation_step.get("output") or {})
        if isinstance(validation_step, dict)
        else {}
    )
    print("\n==== PlanValidation.improved_plan_validation_errors ====")
    print(
        json.dumps(
            (validation_output.get("improved_plan_validation_errors") or []),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n==== PlanValidation.improved_plan_rule_issues ====")
    print(
        json.dumps(
            (validation_output.get("improved_plan_rule_issues") or []),
            ensure_ascii=False,
            indent=2,
        )
    )
    validation_thinking = _require_nonempty_list(
        validation_output.get("thinking_summary"),
        "PlanValidation.thinking_summary",
    )
    _print_thinking("PlanValidation.thinking_summary", validation_thinking)

    export_tool = CDCPlanExportTool()
    export_res = await export_tool.execute(
        plan=final_plan, output_path=args.output_docx
    )
    logger.info(export_res.output or export_res.error or "Export finished.")
    logger.info("End-to-end pipeline completed.")


if __name__ == "__main__":
    asyncio.run(main())
