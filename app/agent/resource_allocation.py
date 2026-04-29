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
        "麻疹": "measles_rubella",
        "风疹": "measles_rubella",
        "麻疹风疹": "measles_rubella",
        "measles": "measles_rubella",
        "rubella": "measles_rubella",
        "百日咳": "pertussis",
        "pertussis": "pertussis",
        "结核": "tuberculosis",
        "结核病": "tuberculosis",
        "肺结核": "tuberculosis",
        "tb": "tuberculosis",
        "tuberculosis": "tuberculosis",
        "登革热": "dengue",
        "dengue": "dengue",
        "手足口": "hand_foot_mouth",
        "手足口病": "hand_foot_mouth",
        "hfmd": "hand_foot_mouth",
        "hand_foot_mouth": "hand_foot_mouth",
        "水痘": "varicella",
        "varicella": "varicella",
        "腮腺炎": "mumps",
        "流行性腮腺炎": "mumps",
        "mumps": "mumps",
        "甲肝": "hepatitis_a",
        "甲型肝炎": "hepatitis_a",
        "hepatitis a": "hepatitis_a",
        "hepatitis_a": "hepatitis_a",
        "食物中毒": "food_poisoning",
        "food_poisoning": "food_poisoning",
    }
    known = set(mapping.values())
    if v in mapping:
        return mapping[v]
    if v in known:
        return v
    return "other"


def _disease_bundle_rules(disease_type: str) -> dict[str, Any]:
    dt = _normalize_disease_type(disease_type)
    base = {
        "core_skus": [
            "mask_surgical",
            "mask_n95",
            "goggles",
            "face_shield",
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
        "measles_rubella": {
            **base,
            "focus": "麻疹/风疹：空气/飞沫传播风险高，强调高等级防护、快速排查、样本采集与免疫补种。",
            "add_skus": [
                "mask_n95",
                "goggles",
                "face_shield",
                "sample_swab",
                "vtm_tube",
                "sample_bag",
                "transport_box",
                "cooler_box",
                "ice_pack",
                "mmr_vaccine",
            ],
        },
        "pertussis": {
            **base,
            "focus": "百日咳：学校/托幼易聚集，强调病例监测、密接管理、实验室检测与宣教。",
            "add_skus": [
                "sample_swab",
                "vtm_tube",
                "pcr_reagent",
                "sample_bag",
                "transport_box",
                "thermometer",
                "warning_sign",
            ],
        },
        "tuberculosis": {
            **base,
            "focus": "结核病：持续传播风险与暴露时长相关，强调呼吸防护、筛查随访与标本采集转运。",
            "add_skus": [
                "mask_n95",
                "sputum_container",
                "biohazard_bag",
                "transport_box",
                "disinfectant",
            ],
        },
        "dengue": {
            **base,
            "focus": "登革热：媒介传播为主，强调防蚊灭蚊、孳生地清理与人群防护。",
            "add_skus": [
                "mosquito_net",
                "mosquito_repellent",
                "larvicide",
                "insecticide",
                "sprayer",
                "warning_sign",
                "megaphone",
            ],
        },
        "hand_foot_mouth": {
            **base,
            "focus": "手足口病：托幼/学校聚集传播，强调手卫生、环境清洁消毒与晨午检。",
            "add_skus": [
                "soap",
                "paper_towel",
                "chlorine_tablet",
                "sprayer",
                "thermometer",
            ],
        },
        "varicella": {
            **base,
            "focus": "水痘：空气传播风险较高，强调隔离观察、通风消毒与免疫补种。",
            "add_skus": [
                "mask_n95",
                "goggles",
                "face_shield",
                "varicella_vaccine",
                "cooler_box",
                "ice_pack",
            ],
        },
        "mumps": {
            **base,
            "focus": "流行性腮腺炎：学校人群易传播，强调病例隔离、健康宣教与免疫补种。",
            "add_skus": [
                "mmr_vaccine",
                "cooler_box",
                "ice_pack",
                "thermometer",
                "warning_sign",
            ],
        },
        "hepatitis_a": {
            **base,
            "focus": "甲型肝炎：经粪口传播，强调饮用水/食品卫生、手卫生与免疫补种。",
            "add_skus": [
                "soap",
                "paper_towel",
                "chlorine_tablet",
                "sprayer",
                "hepatitis_a_vaccine",
                "cooler_box",
                "ice_pack",
            ],
        },
        "food_poisoning": {
            **base,
            "focus": "食物中毒：现场封存与环境消毒、样本采集与转运，消毒与处置耗材为主。",
            "add_skus": ["chlorine_tablet", "sprayer", "biohazard_bag", "sharps_box"],
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
        sku_unit_map = {
            str(i.get("sku") or "").strip(): str(i.get("unit") or "").strip()
            for i in catalog
            if isinstance(i, dict) and str(i.get("sku") or "").strip()
        }
        name_unit_map = {
            str(i.get("name") or "").strip(): str(i.get("unit") or "").strip()
            for i in catalog
            if isinstance(i, dict) and str(i.get("name") or "").strip()
        }
        warehouses = [
            w
            for w in _safe_list(materials_payload.get("warehouses"))
            if isinstance(w, dict)
        ]
        wh_name_map = {
            str(w.get("warehouse_id") or "").strip(): str(w.get("name") or "").strip()
            for w in warehouses
            if str(w.get("warehouse_id") or "").strip()
        }

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
        for d in demands:
            sku = str(d.get("sku") or "").strip()
            unit = str(d.get("unit") or "").strip()
            if sku and (not unit or unit == "unit"):
                u = sku_unit_map.get(sku)
                if u:
                    d["unit"] = u
                    unit = u
            if (not unit or unit == "unit") and d.get("name"):
                nm = str(d.get("name") or "").strip()
                u2 = name_unit_map.get(nm)
                if not u2:
                    for k, v in name_unit_map.items():
                        if nm and k and (nm in k or k in nm):
                            u2 = v
                            break
                if u2:
                    d["unit"] = u2
        demands = demands[:12]

        allocations: List[Dict[str, Any]] = []
        shortages: List[Dict[str, Any]] = []
        for d in demands:
            sku = str(d.get("sku") or "").strip()
            name = str(d.get("name") or "").strip()
            qty = _safe_float(d.get("quantity"), 0.0)
            if qty <= 0:
                continue
            unit = str(d.get("unit") or "").strip() or "unit"
            alloc = await data_api.execute(
                command="materials_allocate",
                sku=sku if sku else None,
                name=name if name else None,
                quantity=qty,
            )
            payload = json.loads(alloc.output) if alloc.output else {}
            allocated_qty = _safe_float(payload.get("allocated_quantity"), 0.0)
            allocs = payload.get("allocations") or []
            if isinstance(allocs, list):
                normalized_allocs = []
                for a in allocs:
                    if not isinstance(a, dict):
                        continue
                    wid = str(a.get("warehouse_id") or "").strip()
                    normalized_allocs.append(
                        {
                            "warehouse_id": wid,
                            "warehouse_name": wh_name_map.get(wid) or wid,
                            "sku": a.get("sku") or payload.get("sku") or sku,
                            "quantity": _safe_float(a.get("quantity"), 0.0),
                        }
                    )
                allocs = normalized_allocs
            allocations.append(
                {
                    "sku": payload.get("sku") or sku,
                    "name": payload.get("name") or name,
                    "unit": unit,
                    "requested_quantity": qty,
                    "allocated_quantity": allocated_qty,
                    "allocations": allocs,
                    "reason": d.get("reason") or "",
                }
            )
            if allocated_qty + 1e-9 < qty:
                shortages.append(
                    {
                        "sku": payload.get("sku") or sku,
                        "name": payload.get("name") or name,
                        "unit": unit,
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
