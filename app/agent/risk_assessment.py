import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from app.agent.base import BaseAgent
from app.llm import LLM
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


@dataclass
class SEIRParams:
    population: int
    initial_infected: int
    initial_exposed: int
    initial_recovered: int
    r0: float
    incubation_days: float
    infectious_days: float
    days: int = 7


@dataclass
class SEIRResult:
    days: int
    s: list[float]
    e: list[float]
    i: list[float]
    r: list[float]
    new_infections: list[float]


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        return float(v)
    except Exception:
        return default


def run_seir(params: SEIRParams) -> SEIRResult:
    n = max(1, int(params.population))
    i0 = max(0, int(params.initial_infected))
    e0 = max(0, int(params.initial_exposed))
    r0_init = max(0, int(params.initial_recovered))
    s0 = max(0, n - i0 - e0 - r0_init)

    r0 = max(0.1, float(params.r0))
    incubation = max(0.5, float(params.incubation_days))
    infectious = max(0.5, float(params.infectious_days))

    sigma = 1.0 / incubation
    gamma = 1.0 / infectious
    beta = r0 * gamma

    days = max(1, int(params.days))
    s = np.zeros(days + 1, dtype=np.float64)
    e = np.zeros(days + 1, dtype=np.float64)
    i = np.zeros(days + 1, dtype=np.float64)
    r = np.zeros(days + 1, dtype=np.float64)
    new_inf = np.zeros(days + 1, dtype=np.float64)

    s[0], e[0], i[0], r[0] = float(s0), float(e0), float(i0), float(r0_init)
    for t in range(days):
        inf_force = beta * s[t] * i[t] / n
        dS = -inf_force
        dE = inf_force - sigma * e[t]
        dI = sigma * e[t] - gamma * i[t]
        dR = gamma * i[t]
        s[t + 1] = max(0.0, s[t] + dS)
        e[t + 1] = max(0.0, e[t] + dE)
        i[t + 1] = max(0.0, i[t] + dI)
        r[t + 1] = max(0.0, r[t] + dR)
        new_inf[t + 1] = max(0.0, sigma * e[t])

    return SEIRResult(
        days=days,
        s=s.round(6).tolist(),
        e=e.round(6).tolist(),
        i=i.round(6).tolist(),
        r=r.round(6).tolist(),
        new_infections=new_inf.round(6).tolist(),
    )


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


def _disease_default_params(disease_type: str) -> dict[str, float]:
    dt = _normalize_disease_type(disease_type)
    defaults = {
        "influenza": {
            "r0": 1.6,
            "incubation_days": 2.0,
            "infectious_days": 4.0,
            "underreport_factor": 1.3,
        },
        "covid19": {
            "r0": 1.8,
            "incubation_days": 3.0,
            "infectious_days": 6.0,
            "underreport_factor": 1.5,
        },
        "norovirus": {
            "r0": 2.2,
            "incubation_days": 1.5,
            "infectious_days": 3.0,
            "underreport_factor": 1.2,
        },
        "measles_rubella": {
            "r0": 10.0,
            "incubation_days": 10.0,
            "infectious_days": 7.0,
            "underreport_factor": 1.2,
        },
        "pertussis": {
            "r0": 5.0,
            "incubation_days": 7.0,
            "infectious_days": 14.0,
            "underreport_factor": 1.2,
        },
        "tuberculosis": {
            "r0": 1.2,
            "incubation_days": 21.0,
            "infectious_days": 30.0,
            "underreport_factor": 1.1,
        },
        "dengue": {
            "r0": 2.3,
            "incubation_days": 6.0,
            "infectious_days": 5.0,
            "underreport_factor": 1.2,
        },
        "hand_foot_mouth": {
            "r0": 2.0,
            "incubation_days": 4.0,
            "infectious_days": 7.0,
            "underreport_factor": 1.2,
        },
        "varicella": {
            "r0": 6.0,
            "incubation_days": 14.0,
            "infectious_days": 7.0,
            "underreport_factor": 1.2,
        },
        "mumps": {
            "r0": 4.0,
            "incubation_days": 16.0,
            "infectious_days": 7.0,
            "underreport_factor": 1.2,
        },
        "hepatitis_a": {
            "r0": 1.4,
            "incubation_days": 28.0,
            "infectious_days": 10.0,
            "underreport_factor": 1.1,
        },
        "food_poisoning": {
            "r0": 1.1,
            "incubation_days": 0.8,
            "infectious_days": 1.5,
            "underreport_factor": 1.1,
        },
        "other": {
            "r0": 1.6,
            "incubation_days": 2.0,
            "infectious_days": 4.0,
            "underreport_factor": 1.3,
        },
    }
    return defaults.get(dt, defaults["other"])


def _estimate_exposed(i0: int, incubation_days: float, infectious_days: float) -> int:
    i0 = max(0, int(i0))
    incubation_days = max(0.5, float(incubation_days))
    infectious_days = max(0.5, float(infectious_days))
    ratio = incubation_days / infectious_days
    return max(1, int(round(max(1.0, i0 * 2.0) * ratio)))


def _summary_fallback(
    *,
    disease_type: str,
    location: str,
    risk_level: str,
    predicted_cases_7d: int,
    growth_factor: float,
) -> str:
    dt = _normalize_disease_type(disease_type)
    dname = {
        "influenza": "流感",
        "covid19": "新冠",
        "norovirus": "诺如病毒",
        "measles_rubella": "麻疹/风疹",
        "pertussis": "百日咳",
        "tuberculosis": "结核病",
        "dengue": "登革热",
        "hand_foot_mouth": "手足口病",
        "varicella": "水痘",
        "mumps": "腮腺炎",
        "hepatitis_a": "甲型肝炎",
        "food_poisoning": "食物中毒",
    }.get(dt, "传染病")
    level_cn = {
        "low": "低",
        "medium": "中",
        "high": "高",
        "extreme": "极高",
    }.get((risk_level or "").strip().lower(), "中")
    loc2 = str(location or "").strip() or "本地"
    pred = max(0, int(predicted_cases_7d))
    gf = float(growth_factor or 0.0)
    action_hint = (
        "建议强化病例监测与信息报告、重点场所卫生管理、风险沟通与医疗救治保障。"
    )
    if dt == "dengue":
        action_hint = "建议同步开展防蚊灭蚊与孳生地清理，强化发热病例监测与就诊指引。"
    if dt == "food_poisoning":
        action_hint = (
            "建议尽快开展就餐史排查与可疑食品封存，强化病例监测、样本采集与风险沟通。"
        )
    return (
        f"综合病例规模与增长趋势，{loc2}{dname}疫情风险评估为{level_cn}风险，"
        f"预计未来7天新增约{pred}例（模型估算，增长系数约{gf:.2f}），{action_hint}"
    )


class RiskAssessmentAgent(BaseAgent):
    name: str = "RiskAssessment"
    description: str = (
        "Risk assessment agent using SEIR simulation and LLM-based interpretation."
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
            "从用户输入中抽取风险评估所需字段，输出严格 JSON（不要输出多余文本）。\n"
            "字段：disease_type(字符串), location(字符串), population(整数), reported_cases(整数), "
            "exposed_cases(整数，可选), recovered_cases(整数，可选), underreport_factor(数值，可选), "
            "r0(数值，可选), incubation_days(数值，可选), infectious_days(数值，可选), days(整数，可选，默认7)。\n"
            "如果缺失：population 默认 10000，reported_cases 默认 10。\n"
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
        population = max(1, _safe_int(extracted.get("population"), 10000))
        reported_cases = max(0, _safe_int(extracted.get("reported_cases"), 10))
        exposed_cases = max(0, _safe_int(extracted.get("exposed_cases"), 0))
        recovered_cases = max(0, _safe_int(extracted.get("recovered_cases"), 0))
        defaults = _disease_default_params(disease_type)
        underreport_factor = _safe_float(
            extracted.get("underreport_factor"), defaults["underreport_factor"]
        )
        r0 = _safe_float(extracted.get("r0"), defaults["r0"])
        incubation_days = _safe_float(
            extracted.get("incubation_days"), defaults["incubation_days"]
        )
        infectious_days = _safe_float(
            extracted.get("infectious_days"), defaults["infectious_days"]
        )
        days = max(1, _safe_int(extracted.get("days"), 7))

        if extracted.get("reported_cases") in (None, "", 0) and location != "未提供":
            data_api = CDCDataAPITool()
            # Try searching by location without specific event type since we separated disease and location
            summary = await data_api.execute(
                command="cases_summary",
                location_contains=location,
            )
            try:
                summary_data = json.loads(summary.output) if summary.output else {}
                series = summary_data.get("series") or []
                if series:
                    last = series[-1]
                    reported_cases = max(
                        reported_cases,
                        _safe_int(last.get("confirmed_cases"), reported_cases),
                    )
            except Exception:
                pass

        i0 = max(0, int(round(reported_cases * max(1.0, underreport_factor))))
        e0 = (
            exposed_cases
            if exposed_cases > 0
            else _estimate_exposed(i0, incubation_days, infectious_days)
        )
        r0_init = recovered_cases

        params = SEIRParams(
            population=population,
            initial_infected=i0,
            initial_exposed=e0,
            initial_recovered=r0_init,
            r0=r0,
            incubation_days=incubation_days,
            infectious_days=infectious_days,
            days=days,
        )
        sim = run_seir(params)
        predicted_7d = int(
            round(sum(sim.new_infections[1 : min(8, len(sim.new_infections))]))
        )
        growth = (sim.i[-1] + sim.r[-1]) / max(1.0, (sim.i[0] + sim.r[0]))
        peak_i = max(sim.i)

        interpret_prompt = (
            "你是疾控风险评估智能体。根据 SEIR 模拟结果与输入信息，给出风险等级与病例预测解释。\n"
            "输出严格 JSON（不要输出多余文本）。\n"
            "字段：risk_level（low|medium|high|extreme），predicted_cases_7d（整数），summary（中文一句话结论），thinking_summary（中文要点，列 3-6 条）。\n"
            "要求：thinking_summary 必须解释 E/I/R 初始化来源（用户提供或估算）以及 disease_type 如何影响参数取值。\n"
            "定级参考（可灵活）：\n"
            "- predicted_cases_7d < 50 且 growth < 1.3 => low\n"
            "- predicted_cases_7d 50-200 或 growth 1.3-2.0 => medium\n"
            "- predicted_cases_7d 200-800 或 growth 2.0-4.0 => high\n"
            "- predicted_cases_7d > 800 或 growth > 4.0 => extreme\n"
        )
        context = {
            "disease_type": disease_type,
            "location": location,
            "population": population,
            "reported_cases": reported_cases,
            "underreport_factor": underreport_factor,
            "params": asdict(params),
            "sim_summary": {
                "predicted_cases_7d": predicted_7d,
                "growth_factor": float(round(growth, 4)),
                "peak_infected": float(round(peak_i, 2)),
            },
            "defaults_by_disease_type": defaults,
        }
        llm_raw = await self.llm.ask(
            messages=[Message.user_message(json.dumps(context, ensure_ascii=False))],
            system_msgs=[Message.system_message(interpret_prompt)],
            stream=False,
            temperature=0.2,
        )

        from app.logger import logger

        logger.info(f"LLM Raw Output (Interpret): {llm_raw}")

        decision = _extract_json(llm_raw)

        ts = decision.get("thinking_summary")
        if not isinstance(ts, list) or not ts:
            refine_prompt = (
                "你是疾控风险评估智能体。请基于给定上下文生成 thinking_summary。\n"
                '输出严格 JSON（不要输出多余文本），格式：{"thinking_summary": ["...", "...", "..."]}。\n'
                "要求：\n"
                "- thinking_summary 不得为空，必须 3-6 条\n"
                "- 必须解释 E/I/R 初始化来源（用户提供或估算）\n"
                "- 必须解释 disease_type 如何影响参数取值\n"
                "- 必须结合 location（场所/地点）说明为什么风险会升/降\n"
                "- 不要输出推理细节，只输出可公开的要点摘要\n"
            )
            refine_ctx = {
                "disease_type": disease_type,
                "location": location,
                "population": population,
                "reported_cases": reported_cases,
                "underreport_factor": underreport_factor,
                "seir_init": {
                    "S0": population - i0 - e0 - r0_init,
                    "E0": e0,
                    "I0": i0,
                    "R0": r0_init,
                },
                "params": asdict(params),
                "sim_summary": {
                    "predicted_cases_7d": predicted_7d,
                    "growth_factor": float(round(growth, 4)),
                    "peak_infected": float(round(peak_i, 2)),
                },
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
                logger.info(f"LLM Raw Output (Refine Attempt {_ + 1}): {refined_raw}")

                refined = _extract_json(refined_raw)
                ts2 = refined.get("thinking_summary")
                if isinstance(ts2, list) and ts2:
                    ts = ts2
                    break

            if not isinstance(ts, list) or not ts:
                raise ValueError("LLM did not return non-empty thinking_summary")
        result = {
            "agent": self.name,
            "input": {
                "disease_type": disease_type,
                "location": location,
                "population": population,
                "reported_cases": reported_cases,
                "exposed_cases": exposed_cases if exposed_cases > 0 else None,
                "recovered_cases": recovered_cases if recovered_cases > 0 else None,
                "underreport_factor": underreport_factor,
                "r0": r0,
                "incubation_days": incubation_days,
                "infectious_days": infectious_days,
                "days": days,
            },
            "seir": {
                "params": asdict(params),
                "result": asdict(sim),
                "derived": {
                    "predicted_cases_7d": predicted_7d,
                    "growth_factor": float(round(growth, 4)),
                    "peak_infected": float(round(peak_i, 2)),
                },
            },
            "assessment": {
                "risk_level": str(decision.get("risk_level") or "medium")
                .strip()
                .lower(),
                "predicted_cases_7d": _safe_int(
                    decision.get("predicted_cases_7d"), predicted_7d
                ),
                "summary": "",
                "thinking_summary": ts,
            },
        }
        risk_level = str(result["assessment"]["risk_level"] or "medium").strip().lower()
        if risk_level not in {"low", "medium", "high", "extreme"}:
            risk_level = "medium"
            result["assessment"]["risk_level"] = risk_level
        pred_out = _safe_int(result["assessment"]["predicted_cases_7d"], predicted_7d)
        if pred_out <= 0:
            pred_out = max(0, int(predicted_7d))
            result["assessment"]["predicted_cases_7d"] = pred_out
        summary_raw = str(decision.get("summary") or "").strip()
        if (
            not summary_raw
            or "未提供风险评估结论" in summary_raw
            or len(summary_raw) < 8
        ):
            summary_raw = _summary_fallback(
                disease_type=disease_type,
                location=location,
                risk_level=risk_level,
                predicted_cases_7d=pred_out,
                growth_factor=growth,
            )
        if summary_raw and summary_raw[-1] not in "。！？””）)":
            summary_raw += "。"
        result["assessment"]["summary"] = summary_raw
        self.memory.add_message(
            Message.assistant_message(json.dumps(result, ensure_ascii=False, indent=2))
        )
        self.state = AgentState.FINISHED
        return json.dumps(result, ensure_ascii=False, indent=2)
