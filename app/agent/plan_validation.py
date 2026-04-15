import json
import re
from typing import Any, Dict, List

from pydantic import ValidationError

from app.agent.base import BaseAgent
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, CDCMeasureLevel, CDCPlanDocument, Message
from app.tool.cdc_guideline_search import CDCGuidelineSearchTool


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


def _normalize_event_type(raw: Any) -> str:
    if raw is None:
        return "other"
    v = str(raw).strip().lower()
    if not v:
        return "other"
    mapping = {
        "covid19": "covid19",
        "covid-19": "covid19",
        "covid": "covid19",
        "新冠": "covid19",
        "新型冠状病毒": "covid19",
        "influenza": "influenza",
        "流感": "influenza",
        "甲流": "influenza",
        "norovirus": "norovirus",
        "诺如": "norovirus",
        "诺如病毒": "norovirus",
        "other": "other",
    }
    return mapping.get(v, str(raw))


def _disease_query_terms(disease_type: str) -> str:
    v = (disease_type or "").strip().lower()
    if v == "covid19":
        return "新冠 COVID-19 SARS-CoV-2 covid19 新型冠状病毒"
    if v == "influenza":
        return "流感 influenza 甲流"
    if v == "norovirus":
        return "诺如 norovirus 诺如病毒"
    return v or "传染病"


def _place_query_terms(location: str) -> str:
    loc = (location or "").strip()
    if not loc:
        return ""
    school_terms = ["学校", "中学", "小学", "大学", "校园", "托幼", "幼儿园"]
    if any(t in loc for t in school_terms):
        return "学校 校园 中学 托幼"
    community_terms = ["社区", "小区", "乡镇", "街道", "村"]
    if any(t in loc for t in community_terms):
        return "社区 小区 乡镇"
    return ""


def _collect_improved_plan_rule_issues(
    improved_plan_obj: Dict[str, Any], retrieved_guidelines: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    allowed_levels = {CDCMeasureLevel.core.value, CDCMeasureLevel.supplementary.value}
    allowed_citations = {
        (str(g.get("source_file")), int(g.get("chunk_id")))
        for g in (retrieved_guidelines or [])
        if isinstance(g, dict)
        and g.get("source_file") is not None
        and g.get("chunk_id") is not None
    }
    measures = improved_plan_obj.get("measures")
    if not isinstance(measures, list):
        return issues
    for idx, m in enumerate(measures):
        if not isinstance(m, dict):
            continue
        level = str(m.get("level") or "").strip()
        if level and level not in allowed_levels:
            issues.append(
                {
                    "type": "invalid_level",
                    "field": f"improved_plan.measures[{idx}].level",
                    "message": f"level 必须为 {sorted(allowed_levels)}",
                    "input": level,
                }
            )
        citations = m.get("citations")
        if level == CDCMeasureLevel.core.value:
            if not isinstance(citations, list) or len(citations) == 0:
                issues.append(
                    {
                        "type": "missing_citation",
                        "field": f"improved_plan.measures[{idx}].citations",
                        "message": "核心措施必须至少包含 1 条规范引用",
                    }
                )
            else:
                for j, c in enumerate(citations):
                    if not isinstance(c, dict):
                        issues.append(
                            {
                                "type": "invalid_citation",
                                "field": f"improved_plan.measures[{idx}].citations[{j}]",
                                "message": "citation 必须为对象",
                                "input": str(c),
                            }
                        )
                        continue
                    if not all(
                        k in c for k in ("source_file", "chunk_id", "score", "excerpt")
                    ):
                        issues.append(
                            {
                                "type": "invalid_citation",
                                "field": f"improved_plan.measures[{idx}].citations[{j}]",
                                "message": "citation 缺少必须字段 source_file/chunk_id/score/excerpt",
                                "input": c,
                            }
                        )
                        continue
                    key = (str(c.get("source_file")), int(c.get("chunk_id")))
                    if allowed_citations and key not in allowed_citations:
                        issues.append(
                            {
                                "type": "citation_not_from_guidelines",
                                "field": f"improved_plan.measures[{idx}].citations[{j}]",
                                "message": "citation 必须从 retrieved_guidelines 中选择（禁止编造）",
                                "input": {"source_file": key[0], "chunk_id": key[1]},
                            }
                        )
    return issues


class PlanValidationAgent(BaseAgent):
    name: str = "PlanValidation"
    description: str = (
        "Validate a plan draft for completeness and compliance with basic rules."
    )
    max_steps: int = 3
    llm: LLM = LLM(config_name="default")

    @staticmethod
    def _collect_rule_issues(plan: CDCPlanDocument) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        if not plan.meta or not plan.meta.title:
            issues.append(
                {
                    "type": "missing_meta",
                    "field": "meta.title",
                    "message": "缺少预案标题",
                }
            )
        if not plan.input or not plan.input.location:
            issues.append(
                {
                    "type": "missing_input",
                    "field": "input.location",
                    "message": "缺少发生地点",
                }
            )
        if plan.input.population <= 0:
            issues.append(
                {
                    "type": "invalid_input",
                    "field": "input.population",
                    "message": "人口数必须大于 0",
                }
            )
        if not plan.risk or not plan.risk.summary:
            issues.append(
                {
                    "type": "missing_risk",
                    "field": "risk.summary",
                    "message": "缺少风险评估结论",
                }
            )
        if not plan.sections:
            issues.append(
                {
                    "type": "missing_sections",
                    "field": "sections",
                    "message": "缺少预案章节结构",
                }
            )

        for idx, m in enumerate(plan.measures or []):
            if m.level == CDCMeasureLevel.core and not m.citations:
                issues.append(
                    {
                        "type": "missing_citation",
                        "field": f"measures[{idx}].citations",
                        "message": "核心措施必须至少包含 1 条规范引用",
                    }
                )
        return issues

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

        extracted = {}
        try:
            direct = json.loads(user_text)
            if isinstance(direct, dict) and isinstance(direct.get("plan"), dict):
                extracted = direct
        except Exception:
            extracted = {}

        if not extracted:
            extract_prompt = (
                "用户会提供一段预案草稿 JSON 或描述。\n"
                "请只输出严格 JSON（不要输出多余文本），字段：plan（对象）。\n"
                "如果用户直接提供 JSON，请原样放入 plan。\n"
            )
            extracted_raw = await self.llm.ask(
                messages=[Message.user_message(user_text)],
                system_msgs=[Message.system_message(extract_prompt)],
                stream=False,
                temperature=0.0,
            )
            extracted = _extract_json(extracted_raw)
        plan_obj = extracted.get("plan")
        if not isinstance(plan_obj, dict):
            self.state = AgentState.FINISHED
            return json.dumps(
                {"error": "无法解析 plan JSON，请直接粘贴 JSON 对象"},
                ensure_ascii=False,
                indent=2,
            )

        validation_errors: List[Dict[str, Any]] = []
        plan: CDCPlanDocument | None = None
        input_obj = (
            plan_obj.get("input") if isinstance(plan_obj.get("input"), dict) else None
        )
        if input_obj is not None:
            input_obj["event_type"] = _normalize_event_type(input_obj.get("event_type"))
        try:
            plan = CDCPlanDocument(**plan_obj)
        except ValidationError as e:
            for err in e.errors():
                validation_errors.append(
                    {
                        "type": err.get("type"),
                        "loc": err.get("loc"),
                        "msg": err.get("msg"),
                        "input": err.get("input"),
                    }
                )

        rule_issues: List[Dict[str, Any]] = []
        if plan is not None:
            rule_issues = self._collect_rule_issues(plan)

        # 1. Gather context from guidelines for knowledge-based validation
        if plan is not None:
            disease_type = plan.input.event_type
            location = plan.input.location
        else:
            disease_type = (
                _normalize_event_type((input_obj or {}).get("event_type"))
                if isinstance(input_obj, dict)
                else "unknown"
            )
            location = (
                str((input_obj or {}).get("location") or "unknown")
                if isinstance(input_obj, dict)
                else "unknown"
            )

        search_tool = CDCGuidelineSearchTool()
        disease_terms = _disease_query_terms(disease_type)
        place_terms = _place_query_terms(location)
        query = f"{disease_terms} {place_terms} 防控 方案 技术 指南 预案 规范 要求"
        search_res = await search_tool.execute(query=query, top_k=8, mode="auto")
        retrieved_guidelines = []
        if search_res.output and isinstance(search_res.output, str):
            try:
                search_data = json.loads(search_res.output)
                retrieved_guidelines = search_data.get("results", [])
            except Exception:
                pass

        improve_prompt = (
            "你是疾控预案校验智能体。根据结构化校验错误(validation_errors)、硬性规则问题(rule_issues)，以及检索到的疾控规范(guidelines)，对预案进行深度业务校验。\n"
            "输出严格 JSON（不要输出多余文本）。\n"
            "字段：valid(布尔), summary(中文一段话), issues(数组), suggestions(数组), thinking_summary(中文要点，列 3-6 条), improved_plan(对象，可选)。\n"
            "要求：\n"
            "1. issues 需结合输入中的 errors/issues，以及是否符合检索到的规范要求来归纳。\n"
            "2. suggestions 给出可执行的修改要点，若存在规范偏离需重点指出。\n"
            "3. 如果输入预案存在缺失或不规范，请在 improved_plan 中给出一份修正后的完整预案 JSON（必须符合 CDCPlanDocument 结构，不要额外字段）。\n"
            "4. improved_plan.measures[].citations 必须是对象数组，每个对象包含 source_file, chunk_id, score, excerpt；并且必须从给定 guidelines 中选择（不要编造）。\n"
            "5. thinking_summary 必须说明如何结合规范知识库发现了问题并给出了建议。\n"
        )
        ctx = {
            "plan_draft": plan_obj,
            "validation_errors": validation_errors,
            "rule_issues": rule_issues,
            "guidelines": retrieved_guidelines,
        }
        llm_raw2 = await self.llm.ask(
            messages=[Message.user_message(json.dumps(ctx, ensure_ascii=False))],
            system_msgs=[Message.system_message(improve_prompt)],
            stream=False,
            temperature=0.2,
        )
        logger.info(f"LLM Raw Output (PlanValidation.Improve): {llm_raw2}")
        improved = _extract_json(llm_raw2)
        ts = improved.get("thinking_summary")
        if not isinstance(ts, list) or not ts:
            refine_prompt = (
                "你是疾控预案校验智能体。请基于校验错误、规范知识库检索结果，输出 thinking_summary。\n"
                '输出严格 JSON（不要输出多余文本），格式：{"thinking_summary": ["...", "...", "..."]}。\n'
                "要求：\n"
                "- thinking_summary 不得为空，必须 3-6 条\n"
                "- 必须分别概括结构性问题与规范依从性问题\n"
                "- 必须说明如何利用知识库指导修复方向\n"
            )
            refine_ctx = {
                "validation_errors": validation_errors,
                "rule_issues": rule_issues,
                "guidelines": retrieved_guidelines,
                "suggestions": improved.get("suggestions") or [],
            }
            ts = None
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
                    f"LLM Raw Output (PlanValidation.Refine Attempt {_ + 1}): {refined_raw}"
                )
                refined = _extract_json(refined_raw)
                ts2 = refined.get("thinking_summary")
                if isinstance(ts2, list) and ts2:
                    ts = ts2
                    break
            if not isinstance(ts, list) or not ts:
                raise ValueError(
                    "LLM did not return non-empty thinking_summary for plan validation"
                )

        valid = bool(plan is not None and not validation_errors and not rule_issues)
        if isinstance(improved.get("valid"), bool):
            valid = improved["valid"]

        improved_plan_obj = improved.get("improved_plan")
        improved_plan_validation_errors: List[Dict[str, Any]] = []
        improved_plan_rule_issues: List[Dict[str, Any]] = []
        if isinstance(improved_plan_obj, dict):
            try:
                CDCPlanDocument(**improved_plan_obj)
            except ValidationError as e:
                improved_plan_validation_errors = [
                    {
                        "type": err.get("type"),
                        "loc": err.get("loc"),
                        "msg": err.get("msg"),
                        "input": err.get("input"),
                    }
                    for err in e.errors()
                ]
            improved_plan_rule_issues = _collect_improved_plan_rule_issues(
                improved_plan_obj, retrieved_guidelines
            )

            if improved_plan_validation_errors or improved_plan_rule_issues:
                repair_prompt = (
                    "你是疾控预案校验智能体。请修复 improved_plan，使其满足结构校验与规则校验。\n"
                    '只输出严格 JSON（不要输出多余文本），格式：{"improved_plan": {...}}。\n'
                    "硬性要求：\n"
                    "1. improved_plan 必须符合 CDCPlanDocument 结构，不要额外字段。\n"
                    "2. improved_plan.measures[].level 只能为 core 或 supplementary。\n"
                    "3. 所有 core 措施必须至少包含 1 条 citations。\n"
                    "4. citations 必须为对象数组，每个对象包含 source_file, chunk_id, score, excerpt。\n"
                    "5. citations 必须从给定 guidelines 中选择（source_file+chunk_id 必须匹配），禁止编造。\n"
                )
                repair_ctx = {
                    "improved_plan": improved_plan_obj,
                    "improved_plan_validation_errors": improved_plan_validation_errors,
                    "improved_plan_rule_issues": improved_plan_rule_issues,
                    "guidelines": retrieved_guidelines,
                }
                for _ in range(2):
                    repaired_raw = await self.llm.ask(
                        messages=[
                            Message.user_message(
                                json.dumps(repair_ctx, ensure_ascii=False)
                            )
                        ],
                        system_msgs=[Message.system_message(repair_prompt)],
                        stream=False,
                        temperature=0.0,
                    )
                    logger.info(
                        f"LLM Raw Output (PlanValidation.Repair Attempt {_ + 1}): {repaired_raw}"
                    )
                    repaired = _extract_json(repaired_raw)
                    candidate = repaired.get("improved_plan")
                    if not isinstance(candidate, dict):
                        continue
                    candidate_validation_errors: List[Dict[str, Any]] = []
                    try:
                        CDCPlanDocument(**candidate)
                    except ValidationError as e:
                        candidate_validation_errors = [
                            {
                                "type": err.get("type"),
                                "loc": err.get("loc"),
                                "msg": err.get("msg"),
                                "input": err.get("input"),
                            }
                            for err in e.errors()
                        ]
                    candidate_rule_issues = _collect_improved_plan_rule_issues(
                        candidate, retrieved_guidelines
                    )
                    if not candidate_validation_errors and not candidate_rule_issues:
                        improved_plan_obj = candidate
                        improved_plan_validation_errors = []
                        improved_plan_rule_issues = []
                        break
                    improved_plan_obj = candidate
                    improved_plan_validation_errors = candidate_validation_errors
                    improved_plan_rule_issues = candidate_rule_issues

        result = {
            "agent": self.name,
            "valid": valid,
            "validation_errors": validation_errors,
            "rule_issues": rule_issues,
            "retrieved_guidelines": retrieved_guidelines,
            "output": {
                "summary": improved.get("summary")
                or ("校验通过。" if valid else "校验未通过。"),
                "issues": improved.get("issues") or [],
                "suggestions": improved.get("suggestions") or [],
                "thinking_summary": ts,
                "improved_plan": improved_plan_obj,
                "improved_plan_validation_errors": improved_plan_validation_errors,
                "improved_plan_rule_issues": improved_plan_rule_issues,
            },
        }
        self.memory.add_message(
            Message.assistant_message(json.dumps(result, ensure_ascii=False, indent=2))
        )
        self.state = AgentState.FINISHED
        return json.dumps(result, ensure_ascii=False, indent=2)
