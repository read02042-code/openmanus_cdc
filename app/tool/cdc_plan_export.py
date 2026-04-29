import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

from app.config import config
from app.schema import CDCMeasureLevel, CDCPlanDocument, CDCPlanMeta, CDCPlanSection
from app.tool.base import BaseTool, ToolResult


class CDCPlanExportTool(BaseTool):
    name: str = "cdc_plan_export"
    description: str = "Export a CDC emergency plan to a .docx or .pdf file."
    parameters: dict = {
        "type": "object",
        "properties": {
            "plan": {
                "description": "Plan object or JSON string that matches CDCPlanDocument schema",
                "anyOf": [{"type": "object"}, {"type": "string"}],
            },
            "output_format": {
                "type": "string",
                "description": "Export format",
                "enum": ["docx", "pdf"],
                "default": "docx",
            },
            "output_path": {
                "type": "string",
                "description": "Output path. If relative, it is created under workspace.",
            },
        },
        "required": ["plan"],
    }

    @staticmethod
    def _normalize_output_path(output_path: Optional[str], output_format: str) -> Path:
        if output_path:
            p = Path(output_path)
            if p.is_absolute():
                return p
            return config.workspace_root / p
        ts = int(time.time())
        suffix = ".pdf" if (output_format or "").strip().lower() == "pdf" else ".docx"
        return config.workspace_root / f"cdc_plan_{ts}{suffix}"

    @staticmethod
    def _normalize_output_format(raw: Any) -> str:
        v = str(raw or "").strip().lower()
        if v in {"pdf", "docx"}:
            return v
        return "docx"

    @staticmethod
    def _parse_plan(plan: Union[str, Dict[str, Any]]) -> CDCPlanDocument:
        if isinstance(plan, str):
            raw = plan.strip()
            if not raw:
                raise ValueError("plan is empty")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("plan JSON must be an object")
            normalized = CDCPlanExportTool._normalize_plan_dict(data)
            return CDCPlanDocument(**normalized)
        if isinstance(plan, dict):
            normalized = CDCPlanExportTool._normalize_plan_dict(plan)
            return CDCPlanDocument(**normalized)
        raise ValueError("plan must be a dict or JSON string")

    @staticmethod
    def _first_nonempty(*values: Any, default: Any = None) -> Any:
        for v in values:
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
        return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            if isinstance(value, bool):
                return default
            if isinstance(value, (int, float)):
                return int(value)
            s = str(value).strip()
            if not s:
                return default
            return int(float(s))
        except Exception:
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            if isinstance(value, bool):
                return default
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).strip()
            if not s:
                return default
            return float(s)
        except Exception:
            return default

    @staticmethod
    def _normalize_event_type(raw: Any) -> str:
        if raw is None:
            return "other"
        v = str(raw).strip()
        if not v:
            return "other"
        mapping = {
            "covid19": "covid19",
            "covid-19": "covid19",
            "covid": "covid19",
            "社区新冠": "covid19",
            "新冠": "covid19",
            "新型冠状病毒": "covid19",
            "influenza": "influenza",
            "学校流感": "influenza",
            "流感": "influenza",
            "甲流": "influenza",
            "norovirus": "norovirus",
            "诺如": "norovirus",
            "诺如病毒": "norovirus",
            "诺如病毒聚集": "norovirus",
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
        v_lower = v.lower()
        if v_lower in mapping:
            return mapping[v_lower]
        if v in mapping:
            return mapping[v]
        known = set(mapping.values())
        if v_lower in known:
            return v_lower
        return "other"

    @staticmethod
    def _normalize_risk_level(raw: Any) -> str:
        if raw is None:
            return "low"
        v = str(raw).strip().lower()
        if not v:
            return "low"
        mapping = {
            "低": "low",
            "低风险": "low",
            "中": "medium",
            "中风险": "medium",
            "高": "high",
            "高风险": "high",
            "极高": "extreme",
            "极高风险": "extreme",
        }
        if v in mapping:
            return mapping[v]
        return v

    @staticmethod
    def _normalize_sections(raw_sections: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_sections, list):
            return []

        def normalize_section(sec: Any) -> Dict[str, Any]:
            if not isinstance(sec, dict):
                return {"title": str(sec), "paragraphs": [], "subsections": []}

            title = CDCPlanExportTool._first_nonempty(
                sec.get("title"),
                sec.get("section_title"),
                sec.get("sectionTitle"),
                sec.get("subsection_title"),
                sec.get("subsectionTitle"),
                default="未命名章节",
            )

            paragraphs_val = CDCPlanExportTool._first_nonempty(
                sec.get("paragraphs"),
                sec.get("paras"),
                sec.get("content"),
                sec.get("text"),
                default=[],
            )
            paragraphs: List[str] = []
            if isinstance(paragraphs_val, list):
                paragraphs = [str(p) for p in paragraphs_val if str(p).strip()]
            elif isinstance(paragraphs_val, str):
                p = paragraphs_val.strip()
                if p:
                    paragraphs = [p]

            raw_subsections = CDCPlanExportTool._first_nonempty(
                sec.get("subsections"),
                sec.get("sub_sections"),
                sec.get("children"),
                default=[],
            )
            subsections: List[Dict[str, Any]] = []
            if isinstance(raw_subsections, list):
                subsections = [normalize_section(s) for s in raw_subsections]

            return {
                "title": str(title),
                "paragraphs": paragraphs,
                "subsections": subsections,
            }

        return [normalize_section(s) for s in raw_sections]

    @staticmethod
    def _normalize_citations(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            out.append(
                {
                    "source_file": str(
                        CDCPlanExportTool._first_nonempty(
                            c.get("source_file"),
                            c.get("source"),
                            c.get("file"),
                            c.get("filename"),
                            default="unknown",
                        )
                    ),
                    "chunk_id": CDCPlanExportTool._as_int(
                        CDCPlanExportTool._first_nonempty(
                            c.get("chunk_id"),
                            c.get("chunkId"),
                            c.get("id"),
                            default=0,
                        ),
                        0,
                    ),
                    "score": CDCPlanExportTool._as_float(
                        CDCPlanExportTool._first_nonempty(
                            c.get("score"),
                            c.get("similarity"),
                            c.get("relevance"),
                            default=0.0,
                        ),
                        0.0,
                    ),
                    "excerpt": str(
                        CDCPlanExportTool._first_nonempty(
                            c.get("excerpt"),
                            c.get("text"),
                            c.get("content"),
                            default="",
                        )
                    ),
                }
            )
        return out

    @staticmethod
    def _normalize_measures(raw_measures: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_measures, list):
            return []
        out: List[Dict[str, Any]] = []
        for m in raw_measures:
            if not isinstance(m, dict):
                continue
            level_raw = CDCPlanExportTool._first_nonempty(
                m.get("level"),
                m.get("measure_level"),
                m.get("type"),
                default="core",
            )
            level = str(level_raw).strip().lower()
            if level in {"核心", "强合规"}:
                level = "core"
            if level in {"补充", "可选", "适配"}:
                level = "supplementary"

            out.append(
                {
                    "title": str(
                        CDCPlanExportTool._first_nonempty(
                            m.get("title"),
                            m.get("measure_title"),
                            m.get("name"),
                            default="未命名措施",
                        )
                    ),
                    "content": str(
                        CDCPlanExportTool._first_nonempty(
                            m.get("content"),
                            m.get("measure_content"),
                            m.get("description"),
                            m.get("text"),
                            default="",
                        )
                    ),
                    "level": level,
                    "citations": CDCPlanExportTool._normalize_citations(
                        CDCPlanExportTool._first_nonempty(
                            m.get("citations"),
                            m.get("references"),
                            m.get("refs"),
                            default=[],
                        )
                    ),
                }
            )
        return out

    @staticmethod
    def _normalize_resources(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            items = raw.get("items")
            if isinstance(items, list):
                return {"items": CDCPlanExportTool._normalize_resource_items(items)}
            if isinstance(raw.get("materials"), list):
                return {
                    "items": CDCPlanExportTool._normalize_resource_items(
                        raw.get("materials")
                    )
                }
        if isinstance(raw, list):
            return {"items": CDCPlanExportTool._normalize_resource_items(raw)}
        return {"items": []}

    @staticmethod
    def _normalize_resource_items(raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            out.append(
                {
                    "name": str(
                        CDCPlanExportTool._first_nonempty(
                            it.get("name"),
                            it.get("item_name"),
                            it.get("resource_name"),
                            default="未知物资",
                        )
                    ),
                    "unit": str(
                        CDCPlanExportTool._first_nonempty(
                            it.get("unit"), default="unit"
                        )
                    ),
                    "quantity": max(
                        0.0,
                        CDCPlanExportTool._as_float(
                            CDCPlanExportTool._first_nonempty(
                                it.get("quantity"),
                                it.get("count"),
                                it.get("amount"),
                                default=0.0,
                            ),
                            0.0,
                        ),
                    ),
                }
            )
        return out

    @staticmethod
    def _normalize_plan_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        meta_in = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        input_in = data.get("input") if isinstance(data.get("input"), dict) else {}
        risk_in = data.get("risk") if isinstance(data.get("risk"), dict) else {}

        meta = {
            "title": str(
                CDCPlanExportTool._first_nonempty(
                    meta_in.get("title"),
                    meta_in.get("plan_title"),
                    meta_in.get("planTitle"),
                    data.get("plan_title"),
                    data.get("planTitle"),
                    data.get("title"),
                    default="疾控应急预案",
                )
            ),
            "jurisdiction": CDCPlanExportTool._first_nonempty(
                meta_in.get("jurisdiction"),
                meta_in.get("unit"),
                data.get("jurisdiction"),
                data.get("unit"),
                data.get("issuing_unit"),
                default=None,
            ),
            "created_at": CDCPlanExportTool._first_nonempty(
                meta_in.get("created_at"),
                meta_in.get("create_time"),
                data.get("created_at"),
                data.get("create_time"),
                default=None,
            ),
        }
        meta["title"] = re.sub(
            r"\s*[（(]\s*(修订版|优化版)\s*[）)]\s*$", "", meta["title"]
        ).strip()
        meta["title"] = (
            re.sub(r"\s*(修订版|优化版)\s*$", "", meta["title"]).strip()
            or meta["title"]
        )

        transmission_in = (
            input_in.get("transmission")
            if isinstance(input_in.get("transmission"), dict)
            else (
                data.get("transmission")
                if isinstance(data.get("transmission"), dict)
                else {}
            )
        )

        event_input = {
            "event_type": CDCPlanExportTool._normalize_event_type(
                CDCPlanExportTool._first_nonempty(
                    input_in.get("event_type"),
                    input_in.get("eventType"),
                    data.get("event_type"),
                    data.get("eventType"),
                    data.get("event"),
                    default="other",
                )
            ),
            "location": str(
                CDCPlanExportTool._first_nonempty(
                    input_in.get("location"),
                    input_in.get("place"),
                    input_in.get("site"),
                    data.get("location"),
                    data.get("place"),
                    data.get("site"),
                    default="未提供",
                )
            ),
            "population": max(
                1,
                CDCPlanExportTool._as_int(
                    CDCPlanExportTool._first_nonempty(
                        input_in.get("population"),
                        input_in.get("region_population"),
                        data.get("population"),
                        data.get("region_population"),
                        default=1,
                    ),
                    1,
                ),
            ),
            "reported_cases": max(
                0,
                CDCPlanExportTool._as_int(
                    CDCPlanExportTool._first_nonempty(
                        input_in.get("reported_cases"),
                        input_in.get("cases"),
                        input_in.get("case_count"),
                        data.get("reported_cases"),
                        data.get("cases"),
                        data.get("case_count"),
                        default=0,
                    ),
                    0,
                ),
            ),
            "report_date": CDCPlanExportTool._first_nonempty(
                input_in.get("report_date"),
                input_in.get("date"),
                data.get("report_date"),
                data.get("date"),
                default=None,
            ),
            "region_profile": (
                str(
                    CDCPlanExportTool._first_nonempty(
                        input_in.get("region_profile"),
                        input_in.get("regionProfile"),
                        input_in.get("key_points"),
                        input_in.get("keyPoints"),
                        data.get("region_profile"),
                        data.get("regionProfile"),
                        data.get("key_points"),
                        data.get("keyPoints"),
                        default="",
                    )
                ).strip()
                or None
            ),
            "transmission": {
                "r0": transmission_in.get("r0"),
                "incubation_days": transmission_in.get("incubation_days"),
                "infectious_days": transmission_in.get("infectious_days"),
            },
        }

        risk = {
            "level": CDCPlanExportTool._normalize_risk_level(
                CDCPlanExportTool._first_nonempty(
                    risk_in.get("level"),
                    risk_in.get("risk_level"),
                    risk_in.get("riskLevel"),
                    data.get("risk_level"),
                    data.get("riskLevel"),
                    default="low",
                )
            ),
            "summary": str(
                CDCPlanExportTool._first_nonempty(
                    risk_in.get("summary"),
                    risk_in.get("analysis"),
                    risk_in.get("risk_summary"),
                    data.get("risk_summary"),
                    data.get("risk_analysis"),
                    default="未提供风险评估结论。",
                )
            ),
            "predicted_cases_7d": CDCPlanExportTool._first_nonempty(
                risk_in.get("predicted_cases_7d"),
                risk_in.get("prediction_7d"),
                data.get("predicted_cases_7d"),
                data.get("prediction_7d"),
                default=None,
            ),
        }

        sections = CDCPlanExportTool._normalize_sections(
            CDCPlanExportTool._first_nonempty(
                data.get("sections"),
                data.get("plan_sections"),
                data.get("outline"),
                default=[],
            )
        )

        measures = CDCPlanExportTool._normalize_measures(
            CDCPlanExportTool._first_nonempty(
                data.get("measures"),
                data.get("control_measures"),
                data.get("actions"),
                default=[],
            )
        )

        resources = CDCPlanExportTool._normalize_resources(
            CDCPlanExportTool._first_nonempty(
                data.get("resources"),
                data.get("stock"),
                data.get("materials"),
                default={"items": []},
            )
        )

        normalized: Dict[str, Any] = {
            "meta": meta,
            "input": event_input,
            "risk": risk,
            "measures": measures,
            "resources": resources,
            "sections": sections,
        }
        return normalized

    @staticmethod
    def _ensure_sections(plan: CDCPlanDocument) -> List[CDCPlanSection]:
        def _strip_leading_enum(text: str) -> str:
            s = str(text or "").strip()
            s = re.sub(r"^\s*\d+\s*[、.．)\]]\s*", "", s)
            s = re.sub(r"^\s*[（(]\s*\d+\s*[）)]\s*", "", s)
            s = re.sub(r"^\s*[一二三四五六七八九十]+\s*[、.．)\]]\s*", "", s)
            s = re.sub(r"^\s*[（(]\s*[一二三四五六七八九十]+\s*[）)]\s*", "", s)
            return s.strip()

        measures_core = [m for m in plan.measures if m.level == CDCMeasureLevel.core]
        measures_supp = [
            m for m in plan.measures if m.level == CDCMeasureLevel.supplementary
        ]
        rp = str(getattr(plan.input, "region_profile", "") or "").strip()
        rp = re.sub(r"[。！？；;]+$", "", rp).strip()

        def _strip_adapt_factors(text: str) -> str:
            s = str(text or "").strip()
            if not s:
                return s
            if rp:
                s2 = re.sub(
                    r"适配因素\s*[:：]\s*" + re.escape(rp) + r"\s*[。！？]?",
                    "",
                    s,
                )
                s = s2 if s2.strip() else s
            s2 = re.sub(r"适配因素\s*[:：][^。！？\n]*[。！？]?", "", s)
            s = s2 if s2.strip() else s
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"^[，,。；;:：、!?！？—-]+\s*", "", s).strip()
            return s

        core_sections: List[CDCPlanSection] = []
        core_sections.append(
            CDCPlanSection(
                title="事件概况",
                paragraphs=[
                    f"事件类型：{getattr(plan.input.event_type, 'value', plan.input.event_type)}",
                    f"发生地点：{plan.input.location}",
                    f"区域人口：{plan.input.population}",
                    f"报告病例数：{plan.input.reported_cases}",
                ],
            )
        )
        core_sections.append(
            CDCPlanSection(
                title="风险评估",
                paragraphs=[
                    f"风险等级：{getattr(plan.risk.level, 'value', plan.risk.level)}",
                    f"评估结论：{plan.risk.summary}",
                ]
                + (
                    [f"未来 7 天病例预测：{plan.risk.predicted_cases_7d}"]
                    if plan.risk.predicted_cases_7d is not None
                    else []
                ),
            )
        )
        if measures_core or measures_supp:
            sub = []
            if measures_core:
                sub.append(
                    CDCPlanSection(
                        title="（一）核心措施（强合规）",
                        paragraphs=[
                            f"{_strip_leading_enum(m.title)}：{_strip_leading_enum(m.content)}"
                            for m in measures_core
                        ],
                    )
                )
            if measures_supp:
                supp_paras: List[str] = [
                    "说明：本节为区域适配措施，应结合属地政策与本地条件（医疗救治能力、学校/养老机构分布、交通可达性、人口结构、物资与检测能力、季节气候等）择优执行或细化后执行。"
                ]
                if rp:
                    supp_paras.append(f"适配因素：{rp}。")
                supp_paras.extend(
                    [
                        f"{_strip_leading_enum(m.title)}：{_strip_leading_enum(_strip_adapt_factors(m.content))}"
                        for m in measures_supp
                    ]
                )
                sub.append(
                    CDCPlanSection(
                        title="（二）补充措施（区域适配）",
                        paragraphs=supp_paras,
                    )
                )
            core_sections.append(CDCPlanSection(title="防控措施", subsections=sub))
        core_sections.append(
            CDCPlanSection(
                title="资源与物资保障",
                paragraphs=(
                    [
                        "说明：以下为系统按“7 天处置周期”估算的物资需求清单（不是库存清单）；实际库存分配与缺口处置见“资源调配与缺口”。",
                        *[
                            f"{_strip_leading_enum(item.name)}：{item.quantity:g}{item.unit}"
                            for item in plan.resources.items
                        ],
                    ]
                    if plan.resources.items
                    else ["暂无资源库存数据。"]
                ),
            )
        )
        signature = CDCPlanSection(
            title="审批与签字",
            paragraphs=["拟稿：", "审核：", "批准：", "（公章占位）"],
        )
        if not plan.sections:

            def _cn_num(n: int) -> str:
                mapping = {
                    1: "一",
                    2: "二",
                    3: "三",
                    4: "四",
                    5: "五",
                    6: "六",
                    7: "七",
                    8: "八",
                    9: "九",
                    10: "十",
                    11: "十一",
                    12: "十二",
                    13: "十三",
                    14: "十四",
                    15: "十五",
                    16: "十六",
                    17: "十七",
                    18: "十八",
                    19: "十九",
                    20: "二十",
                }
                return mapping.get(n, str(n))

            numbered = []
            for i, s in enumerate(core_sections, start=1):
                numbered.append(
                    CDCPlanSection(
                        title=f"{_cn_num(i)}、{s.title}",
                        paragraphs=s.paragraphs,
                        subsections=s.subsections,
                    )
                )
            numbered.append(
                CDCPlanSection(
                    title=f"{_cn_num(len(numbered)+1)}、{signature.title}",
                    paragraphs=signature.paragraphs,
                    subsections=signature.subsections,
                )
            )
            return numbered

        notes = [s for s in plan.sections if s.title == "人工修改说明"]
        rest = [s for s in plan.sections if s.title != "人工修改说明"]

        def _strip_title_prefix(t: str) -> str:
            s = str(t or "").strip()
            s = re.sub(r"^(第)?[一二三四五六七八九十]+[、.]\s*", "", s)
            s = re.sub(r"^\d+[、.]\s*", "", s)
            return s

        def _cn_num(n: int) -> str:
            mapping = {
                1: "一",
                2: "二",
                3: "三",
                4: "四",
                5: "五",
                6: "六",
                7: "七",
                8: "八",
                9: "九",
                10: "十",
                11: "十一",
                12: "十二",
                13: "十三",
                14: "十四",
                15: "十五",
            }
            return mapping.get(n, str(n))

        outline_order = [
            "总则",
            "事件概况",
            "风险评估",
            "应急组织指挥体系",
            "监测、预警与报告",
            "应急响应",
            "防控措施",
            "资源与物资保障",
            "资源调配与缺口",
            "后期处置",
            "保障措施",
            "附则",
            "规范依据（章节来源）",
        ]
        core_map = {s.title: s for s in core_sections}

        rest_map: Dict[str, CDCPlanSection] = {}
        rest_other: List[CDCPlanSection] = []
        for s in rest:
            title = _strip_title_prefix(s.title)
            if not title:
                continue
            if title in outline_order and title not in rest_map:
                rest_map[title] = s
            else:
                rest_other.append(s)

        merged: List[CDCPlanSection] = []
        idx = 1
        for title in outline_order:
            if title in core_map:
                sec = core_map[title]
                merged.append(
                    CDCPlanSection(
                        title=f"{_cn_num(idx)}、{title}",
                        paragraphs=sec.paragraphs,
                        subsections=sec.subsections,
                    )
                )
                idx += 1
                continue
            if title in rest_map:
                sec = rest_map[title]
                merged.append(
                    CDCPlanSection(
                        title=f"{_cn_num(idx)}、{title}",
                        paragraphs=sec.paragraphs,
                        subsections=sec.subsections,
                    )
                )
                idx += 1

        for s in rest_other:
            title = _strip_title_prefix(s.title)
            if not title or title == "人工修改说明":
                continue
            merged.append(
                CDCPlanSection(
                    title=f"{_cn_num(idx)}、{title}",
                    paragraphs=s.paragraphs,
                    subsections=s.subsections,
                )
            )
            idx += 1

        merged.append(
            CDCPlanSection(
                title=f"{_cn_num(idx)}、{signature.title}",
                paragraphs=signature.paragraphs,
                subsections=signature.subsections,
            )
        )
        return notes + merged

    @staticmethod
    def _apply_default_style(doc: Document) -> None:
        def set_style(name: str, font_name: str, size_pt: int, bold: Optional[bool]):
            try:
                st = doc.styles[name]
            except Exception:
                return
            try:
                st.font.name = font_name
                st.font.size = Pt(size_pt)
                if bold is not None:
                    st.font.bold = bold
            except Exception:
                pass
            try:
                st._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
            except Exception:
                pass

        set_style("Normal", "宋体", 11, False)
        set_style("Title", "黑体", 16, True)
        set_style("Heading 1", "黑体", 14, True)
        set_style("Heading 2", "黑体", 12, True)
        set_style("Heading 3", "黑体", 11, True)
        set_style("Heading 4", "黑体", 11, True)

    @staticmethod
    def _write_meta(doc: Document, meta: CDCPlanMeta) -> None:
        doc.add_heading(meta.title, level=0)
        if meta.jurisdiction:
            doc.add_paragraph(f"编制单位：{meta.jurisdiction}")
        if meta.created_at:
            doc.add_paragraph(f"生成时间：{meta.created_at}")
        doc.add_paragraph("")

    @staticmethod
    def _write_section(doc: Document, section: CDCPlanSection, level: int) -> None:
        heading_level = max(1, min(4, level))
        doc.add_heading(section.title, level=heading_level)
        for p in section.paragraphs or []:
            doc.add_paragraph(p)
        for sub in section.subsections or []:
            CDCPlanExportTool._write_section(doc, sub, level + 1)

    @staticmethod
    def _clean_citation_excerpt(excerpt: str) -> str:
        s = str(excerpt or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[（(]\s*来源\s*[:：][^）)]*[）)]", "", s).strip()
        s = re.sub(r"\b\d+(?:\.\d+)+\s*", "", s).strip()
        s = re.sub(
            r"\b\d+\s*(总则|总体要求|工作原则|编制依据|应急响应|后期处置|保障措施|附则)\b",
            "",
            s,
        ).strip()
        if len(s) > 320:
            s = s[:320].rstrip()
        return s

    @staticmethod
    def _write_citations(doc: Document, plan: CDCPlanDocument) -> None:
        by_key: Dict[tuple, Dict[str, Any]] = {}
        for m in plan.measures:
            for c in m.citations:
                key = (
                    c.source_file,
                    CDCPlanExportTool._clean_citation_excerpt(c.excerpt),
                )
                rec = by_key.get(key)
                if rec is None:
                    by_key[key] = {"c": c, "measures": {m.title}, "excerpt": key[1]}
                else:
                    rec["measures"].add(m.title)
        if not by_key:
            return
        doc.add_heading("附：规范依据摘录（引用）", level=1)
        items = []
        for _, rec in by_key.items():
            c = rec["c"]
            measures = sorted(list(rec["measures"]))
            items.append((measures, c, str(rec.get("excerpt") or "")))
        items.sort(key=lambda x: (x[1].source_file or "", x[0][0] if x[0] else ""))
        for i, (measures, c, excerpt) in enumerate(items, 1):
            ms = "、".join([m for m in measures if m]) or "-"
            doc.add_paragraph(f"{i}. 措施：{ms} | 来源：{c.source_file}")
            doc.add_paragraph(
                excerpt or CDCPlanExportTool._clean_citation_excerpt(c.excerpt)
            )

    @staticmethod
    def _build_citations_section(plan: CDCPlanDocument) -> Optional[CDCPlanSection]:
        by_key: Dict[tuple, Dict[str, Any]] = {}
        for m in plan.measures:
            for c in m.citations:
                key = (
                    c.source_file,
                    CDCPlanExportTool._clean_citation_excerpt(c.excerpt),
                )
                rec = by_key.get(key)
                if rec is None:
                    by_key[key] = {"c": c, "measures": {m.title}, "excerpt": key[1]}
                else:
                    rec["measures"].add(m.title)
        if not by_key:
            return None
        items = []
        for _, rec in by_key.items():
            c = rec["c"]
            measures = sorted(list(rec["measures"]))
            items.append((measures, c, str(rec.get("excerpt") or "")))
        items.sort(key=lambda x: (x[1].source_file or "", x[0][0] if x[0] else ""))
        paras: List[str] = []
        for i, (measures, c, excerpt) in enumerate(items, 1):
            ms = "、".join([m for m in measures if m]) or "-"
            paras.append(f"{i}. 措施：{ms} | 来源：{c.source_file}")
            paras.append(
                excerpt or CDCPlanExportTool._clean_citation_excerpt(c.excerpt)
            )
        return CDCPlanSection(title="附：规范依据摘录（引用）", paragraphs=paras)

    @staticmethod
    def _render_html(plan: CDCPlanDocument, sections: List[CDCPlanSection]) -> str:
        def esc(s: Any) -> str:
            return html.escape(str(s or ""), quote=True)

        parts: List[str] = []
        parts.append("<!doctype html>")
        parts.append('<html lang="zh-CN">')
        parts.append("<head>")
        parts.append('<meta charset="utf-8"/>')
        parts.append(
            '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        )
        parts.append(f"<title>{esc(plan.meta.title)}</title>")
        parts.append(
            "<style>"
            "body{font-family:\"Noto Sans CJK SC\",\"Noto Sans SC\",\"Source Han Sans SC\",\"WenQuanYi Micro Hei\",\"Microsoft YaHei\",\"PingFang SC\",\"Hiragino Sans GB\",\"SimSun\",sans-serif;font-size:12pt;line-height:1.6;color:#111;}"
            "h1{font-size:20pt;margin:0 0 10px 0;}"
            "h2{font-size:15pt;margin:18px 0 8px 0;}"
            "h3{font-size:13pt;margin:14px 0 6px 0;}"
            ".meta{margin:0 0 12px 0;}"
            ".p{margin:4px 0;white-space:pre-wrap;}"
            ".sec{page-break-inside:avoid;}"
            "</style>"
        )
        parts.append("</head>")
        parts.append("<body>")
        parts.append(f"<h1>{esc(plan.meta.title)}</h1>")
        parts.append('<div class="meta">')
        if plan.meta.jurisdiction:
            parts.append(
                f'<div class="p">{esc("编制单位：" + plan.meta.jurisdiction)}</div>'
            )
        if plan.meta.created_at:
            parts.append(
                f'<div class="p">{esc("生成时间：" + plan.meta.created_at)}</div>'
            )
        parts.append("</div>")

        def render_section(s: CDCPlanSection, level: int) -> None:
            lv = max(1, min(3, int(level)))
            tag = "h2" if lv == 1 else ("h3" if lv == 2 else "h4")
            parts.append('<div class="sec">')
            parts.append(f"<{tag}>{esc(s.title)}</{tag}>")
            for p in s.paragraphs or []:
                parts.append(f'<div class="p">{esc(p)}</div>')
            for sub in s.subsections or []:
                render_section(sub, level + 1)
            parts.append("</div>")

        for s in sections or []:
            render_section(s, 1)

        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts)

    @staticmethod
    async def _export_pdf_from_html(html_text: str, output_path: Path) -> None:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content(html_text, wait_until="networkidle")
                await page.pdf(
                    path=str(output_path),
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "18mm",
                        "right": "16mm",
                        "bottom": "18mm",
                        "left": "16mm",
                    },
                )
            finally:
                await browser.close()

    @staticmethod
    def _flatten_sections_text(
        plan: CDCPlanDocument, sections: List[CDCPlanSection]
    ) -> List[str]:
        lines: List[str] = []
        title = str(plan.meta.title or "").strip()
        if title:
            lines.append(title)
        if plan.meta.jurisdiction:
            lines.append(f"编制单位：{plan.meta.jurisdiction}")
        if plan.meta.created_at:
            lines.append(f"生成时间：{plan.meta.created_at}")
        if lines:
            lines.append("")

        def walk(sec: CDCPlanSection) -> None:
            t = str(sec.title or "").strip()
            if t:
                lines.append(t)
            for p in sec.paragraphs or []:
                s = str(p or "").strip()
                if s:
                    lines.append(s)
            lines.append("")
            for sub in sec.subsections or []:
                walk(sub)

        for s in sections or []:
            walk(s)
        while lines and not str(lines[-1]).strip():
            lines.pop()
        return lines

    @staticmethod
    def _export_pdf_with_pillow(
        *, plan: CDCPlanDocument, sections: List[CDCPlanSection], output_path: Path
    ) -> None:
        raise RuntimeError("pillow_not_available")

    @staticmethod
    def _export_pdf_minimal_text(
        *, plan: CDCPlanDocument, sections: List[CDCPlanSection], output_path: Path
    ) -> None:
        page_w = 595
        page_h = 842
        margin_x = 50
        margin_y = 56
        font_size = 12
        line_h = 16
        max_lines_per_page = max(1, int((page_h - margin_y * 2) // line_h))
        max_units = 52

        def units(s: str) -> float:
            total = 0.0
            for ch in s:
                o = ord(ch)
                if o <= 0x7F:
                    total += 0.55
                else:
                    total += 1.0
            return total

        def wrap(s: str) -> List[str]:
            t = str(s or "").strip()
            if not t:
                return [""]
            out: List[str] = []
            cur = ""
            for ch in t:
                cand = cur + ch
                if units(cand) <= max_units:
                    cur = cand
                    continue
                if cur:
                    out.append(cur)
                    cur = ch
                else:
                    out.append(cand)
                    cur = ""
            if cur or not out:
                out.append(cur)
            return out

        raw_lines = CDCPlanExportTool._flatten_sections_text(plan, sections)
        lines: List[str] = []
        for s in raw_lines:
            if not str(s or "").strip():
                lines.append("")
                continue
            lines.extend(wrap(s))

        pages: List[List[str]] = []
        cur: List[str] = []
        for s in lines:
            cur.append(s)
            if len(cur) >= max_lines_per_page:
                pages.append(cur)
                cur = []
        if cur:
            pages.append(cur)
        if not pages:
            pages = [[""]]

        def pdf_hex_text(s: str) -> str:
            b = str(s or "").encode("utf-16be", errors="ignore")
            return b.hex().upper()

        objects: List[bytes] = []

        def add_obj(payload: bytes) -> int:
            objects.append(payload)
            return len(objects)

        font_obj = add_obj(
            b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light "
            b"/Encoding /UniGB-UCS2-H /DescendantFonts [6 0 R] >>"
        )
        cid_font_obj = add_obj(
            b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> "
            b"/DW 1000 >>"
        )

        content_obj_ids: List[int] = []
        page_obj_ids: List[int] = []

        for page_lines in pages:
            y0 = page_h - margin_y
            x0 = margin_x
            parts: List[str] = []
            parts.append("BT")
            parts.append(f"/F1 {font_size} Tf")
            parts.append(f"{line_h} TL")
            parts.append(f"1 0 0 1 {x0} {y0} Tm")
            for s in page_lines:
                if not str(s or "").strip():
                    parts.append("T*")
                    continue
                hx = pdf_hex_text(s)
                parts.append(f"<{hx}> Tj")
                parts.append("T*")
            parts.append("ET")
            stream = ("\n".join(parts)).encode("utf-8")
            content_obj_ids.append(
                add_obj(
                    b"<< /Length "
                    + str(len(stream)).encode("ascii")
                    + b" >>\nstream\n"
                    + stream
                    + b"\nendstream"
                )
            )

        pages_kids = []
        for content_id in content_obj_ids:
            page_payload = (
                b"<< /Type /Page /Parent 2 0 R "
                b"/MediaBox [0 0 "
                + str(page_w).encode("ascii")
                + b" "
                + str(page_h).encode("ascii")
                + b"] "
                b"/Resources << /Font << /F1 "
                + str(font_obj).encode("ascii")
                + b" 0 R >> >> "
                b"/Contents " + str(content_id).encode("ascii") + b" 0 R >>"
            )
            page_obj_ids.append(add_obj(page_payload))
            pages_kids.append(f"{page_obj_ids[-1]} 0 R")

        pages_obj_payload = (
            b"<< /Type /Pages /Kids ["
            + (" ".join(pages_kids)).encode("ascii")
            + b"] /Count "
            + str(len(page_obj_ids)).encode("ascii")
            + b" >>"
        )
        pages_obj_id = 2
        if pages_obj_id != len(objects) + 1:
            objects.insert(0, b"")
            font_obj += 1
            cid_font_obj += 1
            content_obj_ids[:] = [x + 1 for x in content_obj_ids]
            page_obj_ids[:] = [x + 1 for x in page_obj_ids]
            pages_kids = [f"{int(x.split()[0]) + 1} 0 R" for x in pages_kids]
        objects[1] = pages_obj_payload
        catalog_obj_id = add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

        objects[font_obj - 1] = objects[font_obj - 1].replace(
            b"[6 0 R]", f"[{cid_font_obj} 0 R]".encode("ascii")
        )

        out = bytearray()
        out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, payload in enumerate(objects, 1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode("ascii"))
            out.extend(payload)
            out.extend(b"\nendobj\n")
        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        out.extend(b"trailer\n")
        out.extend(
            f"<< /Size {len(objects)+1} /Root {catalog_obj_id} 0 R >>\n".encode("ascii")
        )
        out.extend(b"startxref\n")
        out.extend(f"{xref_pos}\n".encode("ascii"))
        out.extend(b"%%EOF\n")
        output_path.write_bytes(bytes(out))

    async def execute(self, **kwargs) -> ToolResult:
        try:
            plan = self._parse_plan(kwargs.get("plan"))
        except Exception as e:
            return ToolResult(error=str(e))

        output_format = self._normalize_output_format(kwargs.get("output_format"))
        output_path = self._normalize_output_path(
            kwargs.get("output_path"), output_format
        )
        if output_format == "pdf" and output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")
        if output_format == "docx" and output_path.suffix.lower() != ".docx":
            output_path = output_path.with_suffix(".docx")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = self._ensure_sections(plan)
        citations_sec = self._build_citations_section(plan)
        export_sections = sections + ([citations_sec] if citations_sec else [])
        if output_format == "pdf":
            html_text = self._render_html(plan, export_sections)
            try:
                await self._export_pdf_from_html(html_text, output_path)
            except Exception as e:
                msg = str(e)
                hint = ""
                if any(
                    k in msg.lower()
                    for k in [
                        "executable doesn't exist",
                        "browser",
                        "chromium",
                        "playwright",
                    ]
                ):
                    hint = "（提示：可能缺少浏览器驱动，可尝试运行：python -m playwright install chromium）"
                try:
                    self._export_pdf_minimal_text(
                        plan=plan,
                        sections=export_sections,
                        output_path=output_path,
                    )
                    return self.success_response(
                        {
                            "output_path": str(output_path),
                            "title": plan.meta.title,
                            "note": f"pdf_render_fallback: minimal{hint}",
                        }
                    )
                except Exception as e2:
                    return ToolResult(
                        error=f"pdf_export_failed: {msg}{hint}; minimal_fallback_failed: {e2}"
                    )
            return self.success_response(
                {"output_path": str(output_path), "title": plan.meta.title}
            )

        doc = Document()
        try:
            self._apply_default_style(doc)
        except Exception:
            pass

        self._write_meta(doc, plan.meta)
        for s in sections:
            self._write_section(doc, s, level=1)
        self._write_citations(doc, plan)

        try:
            doc.save(str(output_path))
        except PermissionError:
            ts = int(time.time())
            fallback = output_path.with_name(
                f"{output_path.stem}_{ts}{output_path.suffix}"
            )
            doc.save(str(fallback))
            output_path = fallback
        return self.success_response(
            {"output_path": str(output_path), "title": plan.meta.title}
        )
