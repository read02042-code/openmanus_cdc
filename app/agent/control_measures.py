import json
import re
from typing import Any, Dict, List

from app.agent.base import BaseAgent
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message
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


def _disease_query_templates(
    disease_type: str, location: str, risk_level: str
) -> list[str]:
    dt = _normalize_disease_type(disease_type)
    loc = location if location != "未提供" else ""
    rl = (risk_level or "medium").lower()

    # Add location to base queries if provided
    loc_prefix = f"{loc} " if loc else ""
    base = [
        f"{loc_prefix}{dt} {rl} 防控 措施 规范",
        f"{loc_prefix}{dt} 监测 报告 流调 密接 管理",
    ]

    templates = {
        "influenza": base
        + [
            f"{loc_prefix}流感 聚集性 疫情 处置 停课 阈值 规范",
            f"{loc_prefix}晨午检 缺课追踪 病例管理 健康宣教 规范",
            "通风 消毒 防控 措施 规范",
        ],
        "covid19": base
        + [
            f"{loc_prefix}新冠 监测 报告 密接 管理 风险沟通 规范",
            "医疗机构 预检分诊 发热门诊 核酸 抗原 规范",
            f"{loc_prefix}重点场所 消毒 通风 个人防护 指南",
        ],
        "norovirus": base
        + [
            f"{loc_prefix}诺如 病毒 聚集 处置 流调 采样 检测 规范",
            "呕吐物 处置 环境 消毒 餐饮 饮水 卫生 指南",
            f"{loc_prefix}诺如 胃肠炎 聚集 防控 措施 规范",
        ],
        "other": base + ["公共卫生事件 应急 处置 规范", "消毒 个人防护 风险沟通 规范"],
    }
    return templates.get(dt, templates["other"])


class ControlMeasuresAgent(BaseAgent):
    name: str = "ControlMeasures"
    description: str = (
        "Generate control measures based on guideline retrieval and LLM reasoning."
    )
    max_steps: int = 3
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
            "从用户输入中抽取防控措施生成所需字段，输出严格 JSON（不要输出多余文本）。\n"
            "字段：disease_type(字符串), location(字符串), risk_level(low|medium|high|extreme), key_points(字符串，可选)。\n"
            "如果缺失：disease_type=other, location=未提供, risk_level=medium。\n"
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
        key_points = str(extracted.get("key_points") or "").strip()

        queries = _disease_query_templates(disease_type, location, risk_level)
        if key_points:
            queries.insert(
                0, f"{location} {disease_type} {risk_level} {key_points} 规范"
            )

        search_tool = CDCGuidelineSearchTool()
        retrieved: List[Dict[str, Any]] = []
        for q in queries[:5]:
            r = await search_tool.execute(query=q, top_k=5, mode="auto")
            try:
                payload = json.loads(r.output) if r.output else {}
                retrieved.extend(_safe_list(payload.get("results")))
            except Exception:
                continue

        seen = set()
        unique: List[Dict[str, Any]] = []
        for item in retrieved:
            key = (item.get("source_file"), item.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        unique = unique[:10]

        generate_prompt = (
            "你是疾控防控措施智能体。给定疫情类型、风险等级与规范检索片段，生成可执行的防控措施。\n"
            "输出严格 JSON（不要输出多余文本）。\n"
            "字段：measures（数组）与 thinking_summary（中文要点，列 3-6 条）。\n"
            "每条措施结构：title(字符串), content(字符串), level(core|supplementary), citations(数组)。\n"
            "citations 结构：source_file, chunk_id, score, excerpt。\n"
            "硬约束：所有 core 措施 citations 至少 1 条。\n"
            "要求：每条措施 content 需要包含“适用场景/触发条件”（一句话即可），并结合 location 和 disease_type 的特性。\n"
        )
        context = {
            "disease_type": disease_type,
            "location": location,
            "risk_level": risk_level,
            "guidelines": unique,
        }
        llm_raw = await self.llm.ask(
            messages=[Message.user_message(json.dumps(context, ensure_ascii=False))],
            system_msgs=[Message.system_message(generate_prompt)],
            stream=False,
            temperature=0.3,
        )
        logger.info(f"LLM Raw Output (ControlMeasures.Generate): {llm_raw}")
        plan = _extract_json(llm_raw)

        ts = plan.get("thinking_summary")
        if not isinstance(ts, list) or not ts:
            refine_prompt = (
                "你是疾控防控措施智能体。请基于输入场景、检索到的规范片段、以及生成的措施，输出 thinking_summary。\n"
                '输出严格 JSON（不要输出多余文本），格式：{"thinking_summary": ["...", "...", "..."]}。\n'
                "要求：\n"
                "- thinking_summary 不得为空，必须 3-6 条\n"
                "- 必须说明 disease_type 与 location 对措施选择的影响\n"
                "- 必须说明引用规范如何支撑核心措施\n"
                "- 不要输出推理细节，只输出可公开的要点摘要\n"
            )
            refine_ctx = {
                "disease_type": disease_type,
                "location": location,
                "risk_level": risk_level,
                "guidelines": unique,
                "measures": (
                    plan.get("measures")
                    if isinstance(plan.get("measures"), list)
                    else []
                ),
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
                    f"LLM Raw Output (ControlMeasures.Refine Attempt {_ + 1}): {refined_raw}"
                )
                refined = _extract_json(refined_raw)
                ts2 = refined.get("thinking_summary")
                if isinstance(ts2, list) and ts2:
                    ts = ts2
                    break
            if not isinstance(ts, list) or not ts:
                raise ValueError("LLM did not return non-empty thinking_summary")

        measures = _safe_list(plan.get("measures"))
        fallback_citation = unique[:1]
        fixed_measures = []
        for m in measures:
            if not isinstance(m, dict):
                continue
            level = str(m.get("level") or "core").lower()
            citations = _safe_list(m.get("citations"))
            if level == "core" and not citations and fallback_citation:
                citations = [
                    {
                        "source_file": fallback_citation[0].get(
                            "source_file", "unknown"
                        ),
                        "chunk_id": fallback_citation[0].get("chunk_id", 0),
                        "score": fallback_citation[0].get("score", 0.0),
                        "excerpt": fallback_citation[0].get("excerpt", ""),
                    }
                ]
            fixed_measures.append(
                {
                    "title": str(m.get("title") or "未命名措施"),
                    "content": str(m.get("content") or ""),
                    "level": "supplementary" if level == "supplementary" else "core",
                    "citations": citations,
                }
            )

        result = {
            "agent": self.name,
            "input": {
                "disease_type": disease_type,
                "location": location,
                "risk_level": risk_level,
                "key_points": key_points,
            },
            "retrieved_guidelines": unique,
            "output": {
                "measures": fixed_measures,
                "thinking_summary": ts,
            },
        }
        self.memory.add_message(
            Message.assistant_message(json.dumps(result, ensure_ascii=False, indent=2))
        )
        self.state = AgentState.FINISHED
        return json.dumps(result, ensure_ascii=False, indent=2)
