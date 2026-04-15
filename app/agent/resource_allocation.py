import json
import re
from typing import Any, Dict, List

from app.agent.base import BaseAgent
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message
from app.tool.cdc_data_api import CDCDataAPITool


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def _safe_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        return float(v)
    except Exception:
        return default


def _normalize_disease_type(disease_type: str) -> str:
    v = (disease_type or "").strip().lower()
    if not v:
        return "other"
    mapping = {
        "流感": "influenza",
        "influenza": "influenza",
        "甲流": "influenza",
        "新冠": "covid19",
        "新型冠状病毒": "covid19",
        "covid19": "covid19",
        "covid-19": "covid19",
        "covid": "covid19",
        "诺如": "norovirus",
        "诺如病毒": "norovirus",
        "胃肠炎": "norovirus",
        "norovirus": "norovirus",
    }
    return mapping.get(v, v)


def _disease_bundle_rules(disease_type: str) -> dict[str, Any]:
    dt = _normalize_disease_type(disease_type)
    base = {
        "core_skus": [
            "mask_surgical",
            "mask_n95",
            "gloves",
            "protective_suit",
            "isolation_gown",
            "hand_sanitizer",
            "disinfectant",
            "biohazard_bag",
            "sharps_box",
        ],
        "notes": "基础包适用于大多数呼吸道/聚集事件，强调个人防护、消毒与医疗废物处置。",
    }
    bundles = {
        "influenza": {
            **base,
            "focus": "流感：晨午检/缺课追踪/通风消毒，防护与消毒耗材为主。",
            "add_skus": ["thermometer", "temp_gun", "chlorine_tablet", "sprayer"],
        },
        "covid19": {
            **base,
            "focus": "新冠：监测报告/密接管理/检测能力，采样与检测耗材为主。",
            "add_skus": [
                "antigen_test",
                "pcr_reagent",
                "sample_swab",
                "vtm_tube",
                "sample_bag",
                "transport_box",
                "cooler_box",
                "ice_pack",
            ],
        },
        "norovirus": {
            **base,
            "focus": "诺如：呕吐物与环境消毒，处置与消毒耗材为主。",
            "add_skus": ["chlorine_tablet", "sprayer", "shoe_cover"],
        },
        "other": base,
    }
    return bundles.get(dt, bundles["other"])


class ResourceAllocationAgent(BaseAgent):
    name: str = "ResourceAllocation"
    description: str = (
        "Allocate materials based on scenario demand using CDC data API and LLM reasoning."
    )
    max_steps: int = 4
    llm: LLM = LLM(config_name="default")

    async def step(self) -> str:
        user_text = ""
        for msg in reversed(self.memory.messages):
            if msg.role == "user" and msg.content:
                user_text = msg.content
                break
        if not user_text:
            self.state = AgentState.FINISHED
            return json.dumps(
                {"error": "missing user request"}, ensure_ascii=False, indent=2
            )

        extract_prompt = (
            "从用户输入中抽取资源调配所需字段，输出严格 JSON（不要输出多余文本）。\n"
            "字段：disease_type(字符串), location(字符串), risk_level(low|medium|high|extreme), population(整数，可选), "
            "cases(整数，可选), days(整数，可选，默认 7)。\n"
        )
        extracted_raw = await self.llm.ask(
            messages=[Message.user_message(user_text)],
            system_msgs=[Message.system_message(extract_prompt)],
            stream=False,
            temperature=0.0,
        )
        extracted = _extract_json(extracted_raw)

        disease_type = _normalize_disease_type(
            str(extracted.get("disease_type") or "other")
        )
        location = str(extracted.get("location") or "未提供")
        risk_level = str(extracted.get("risk_level") or "medium").lower()
        population = int(extracted.get("population") or 0)
        cases = int(extracted.get("cases") or 0)
        days = int(extracted.get("days") or 7)

        data_api = CDCDataAPITool()
        materials_list = await data_api.execute(command="materials_list")
        materials_payload = (
            json.loads(materials_list.output) if materials_list.output else {}
        )

        catalog = [
            {
                "sku": i.get("sku"),
                "name": i.get("name"),
                "unit": i.get("unit"),
                "category": i.get("category"),
            }
            for i in _safe_list(materials_payload.get("items"))
        ]

        bundle = _disease_bundle_rules(disease_type)

        demand_prompt = (
            "你是疾控资源调配智能体。根据事件、风险等级、人口与病例数，生成物资需求清单。\n"
            "你必须从给定物资目录中选择，输出严格 JSON（不要输出多余文本）。\n"
            "字段：demands（数组）与 thinking_summary（中文要点，列 3-6 条）。\n"
            "每条 demand：sku(字符串), name(字符串), quantity(数值), reason(字符串)。\n"
            "要求：必须结合 disease_type 和 location 的特性；优先覆盖推荐物资包中的 skus；并解释数量与 cases/days 的关系。\n"
        )
        context = {
            "disease_type": disease_type,
            "location": location,
            "risk_level": risk_level,
            "population": population,
            "cases": cases,
            "days": days,
            "catalog": catalog,
            "recommended_bundle": bundle,
        }
        llm_raw = await self.llm.ask(
            messages=[Message.user_message(json.dumps(context, ensure_ascii=False))],
            system_msgs=[Message.system_message(demand_prompt)],
            stream=False,
            temperature=0.3,
        )
        logger.info(f"LLM Raw Output (ResourceAllocation.Demands): {llm_raw}")
        plan = _extract_json(llm_raw)

        demand_ts = plan.get("thinking_summary")
        if not isinstance(demand_ts, list) or not demand_ts:
            refine_prompt = (
                "你是疾控资源调配智能体。请基于输入场景、推荐物资包与生成的需求清单，输出 thinking_summary。\n"
                '输出严格 JSON（不要输出多余文本），格式：{"thinking_summary": ["...", "...", "..."]}。\n'
                "要求：\n"
                "- thinking_summary 不得为空，必须 3-6 条\n"
                "- 必须说明 disease_type 与 location 如何影响物资包与数量\n"
                "- 必须说明数量与 cases/days 的关系\n"
                "- 不要输出推理细节，只输出可公开的要点摘要\n"
            )
            refine_ctx = {
                "disease_type": disease_type,
                "location": location,
                "risk_level": risk_level,
                "population": population,
                "cases": cases,
                "days": days,
                "recommended_bundle": bundle,
                "demands": (
                    plan.get("demands") if isinstance(plan.get("demands"), list) else []
                ),
            }
            demand_ts = None
            for _ in range(3):
                refined_raw = await self.llm.ask(
                    messages=[
                        Message.user_message(json.dumps(refine_ctx, ensure_ascii=False))
                    ],
                    system_msgs=[Message.system_message(refine_prompt)],
                    stream=False,
                    temperature=0.0,
                )
                logger.info(
                    f"LLM Raw Output (ResourceAllocation.Demands.Refine Attempt {_ + 1}): {refined_raw}"
                )
                refined = _extract_json(refined_raw)
                ts2 = refined.get("thinking_summary")
                if isinstance(ts2, list) and ts2:
                    demand_ts = ts2
                    break
            if not isinstance(demand_ts, list) or not demand_ts:
                raise ValueError(
                    "LLM did not return non-empty thinking_summary for demands"
                )

        demands = [d for d in _safe_list(plan.get("demands")) if isinstance(d, dict)]
        demands = demands[:12]

        allocations: List[Dict[str, Any]] = []
        shortages: List[Dict[str, Any]] = []
        for d in demands:
            sku = str(d.get("sku") or "").strip()
            name = str(d.get("name") or "").strip()
            qty = _safe_float(d.get("quantity"), 0.0)
            if qty <= 0:
                continue
            alloc = await data_api.execute(
                command="materials_allocate",
                sku=sku if sku else None,
                name=name if name else None,
                quantity=qty,
            )
            payload = json.loads(alloc.output) if alloc.output else {}
            allocated_qty = _safe_float(payload.get("allocated_quantity"), 0.0)
            allocations.append(
                {
                    "sku": payload.get("sku") or sku,
                    "name": payload.get("name") or name,
                    "requested_quantity": qty,
                    "allocated_quantity": allocated_qty,
                    "allocations": payload.get("allocations") or [],
                    "reason": d.get("reason") or "",
                }
            )
            if allocated_qty + 1e-9 < qty:
                shortages.append(
                    {
                        "sku": payload.get("sku") or sku,
                        "name": payload.get("name") or name,
                        "shortage": float(qty - allocated_qty),
                    }
                )

        narrative_prompt = (
            "你是疾控资源调配智能体。基于物资需求、分配结果与缺口，输出调拨/申请方案。\n"
            "输出严格 JSON（不要输出多余文本）。\n"
            "字段：summary（中文一段话）、actions（数组，列出调拨/采购/借用/替代方案）、thinking_summary（中文要点，列 3-6 条）。\n"
        )
        narrative_ctx = {
            "disease_type": disease_type,
            "location": location,
            "risk_level": risk_level,
            "demands": demands,
            "allocations": allocations,
            "shortages": shortages,
        }
        llm_raw2 = await self.llm.ask(
            messages=[
                Message.user_message(json.dumps(narrative_ctx, ensure_ascii=False))
            ],
            system_msgs=[Message.system_message(narrative_prompt)],
            stream=False,
            temperature=0.3,
        )
        logger.info(f"LLM Raw Output (ResourceAllocation.Narrative): {llm_raw2}")
        narrative = _extract_json(llm_raw2)

        narrative_ts = narrative.get("thinking_summary")
        if not isinstance(narrative_ts, list) or not narrative_ts:
            refine_prompt = (
                "你是疾控资源调配智能体。请基于分配结果与缺口，输出 thinking_summary。\n"
                '输出严格 JSON（不要输出多余文本），格式：{"thinking_summary": ["...", "...", "..."]}。\n'
                "要求：\n"
                "- thinking_summary 不得为空，必须 3-6 条\n"
                "- 必须说明分配策略（调拨/申请/替代）的依据\n"
                "- 必须提到 shortages（若存在）及其处置建议\n"
                "- 不要输出推理细节，只输出可公开的要点摘要\n"
            )
            refine_ctx = {
                "disease_type": disease_type,
                "location": location,
                "risk_level": risk_level,
                "demands": demands,
                "allocation_result": {
                    "allocations": allocations,
                    "shortages": shortages,
                },
                "summary": narrative.get("summary"),
                "actions": narrative.get("actions"),
            }
            narrative_ts = None
            for _ in range(3):
                refined_raw = await self.llm.ask(
                    messages=[
                        Message.user_message(json.dumps(refine_ctx, ensure_ascii=False))
                    ],
                    system_msgs=[Message.system_message(refine_prompt)],
                    stream=False,
                    temperature=0.0,
                )
                logger.info(
                    f"LLM Raw Output (ResourceAllocation.Narrative.Refine Attempt {_ + 1}): {refined_raw}"
                )
                refined = _extract_json(refined_raw)
                ts2 = refined.get("thinking_summary")
                if isinstance(ts2, list) and ts2:
                    narrative_ts = ts2
                    break
            if not isinstance(narrative_ts, list) or not narrative_ts:
                raise ValueError(
                    "LLM did not return non-empty thinking_summary for allocation narrative"
                )

        result = {
            "agent": self.name,
            "input": {
                "disease_type": disease_type,
                "location": location,
                "risk_level": risk_level,
                "population": population,
                "cases": cases,
                "days": days,
            },
            "demands": demands,
            "allocation_result": {
                "allocations": allocations,
                "shortages": shortages,
            },
            "output": {
                "summary": narrative.get("summary") or "已生成资源调配建议。",
                "actions": narrative.get("actions") or [],
                "thinking_summary": narrative_ts,
                "demands_thinking_summary": demand_ts,
            },
        }
        self.memory.add_message(
            Message.assistant_message(json.dumps(result, ensure_ascii=False, indent=2))
        )
        self.state = AgentState.FINISHED
        return json.dumps(result, ensure_ascii=False, indent=2)
