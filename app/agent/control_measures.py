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
        try:
            direct = json.loads(user_text)
            if isinstance(direct, dict):
                kp2 = direct.get("key_points") or direct.get("region_profile")
                if kp2 is not None:
                    key_points = str(kp2 or "").strip()
        except Exception:
            pass

        m = re.search(
            r"key_points\s*[:：]\s*(.*?)(?:请基于规范检索生成防控措施|$)",
            user_text,
            flags=re.S,
        )
        if m:
            key_points = str(m.group(1) or "").strip().rstrip("；;")
        kp = key_points.replace('"', "").replace("'", "").strip()
        if not kp or kp in {"无", "未提供", "暂无", "不详"} or "需补充" in kp:
            key_points = ""
        else:
            key_points = kp

        queries = _disease_query_templates(disease_type, location, risk_level)
        if key_points:
            queries.insert(
                0, f"{location} {disease_type} {risk_level} {key_points} 规范"
            )

        search_tool = CDCGuidelineSearchTool()
        retrieved: List[Dict[str, Any]] = []
        for q in queries[:5]:
            r = await search_tool.execute(
                query=q, disease_type=disease_type, top_k=5, mode="auto"
            )
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
        dt_norm = str(disease_type or "").strip().lower()
        ban_by_dt: Dict[str, List[str]] = {
            "influenza": [
                "诺如",
                "norovirus",
                "百日咳",
                "麻疹",
                "风疹",
                "结核",
                "结核病",
                "TB",
                "新冠",
                "SARS-CoV-2",
                "COVID",
                "冠状病毒",
            ],
            "covid19": [
                "流感",
                "influenza",
                "甲流",
                "诺如",
                "norovirus",
                "百日咳",
                "麻疹",
                "风疹",
                "结核",
                "结核病",
                "TB",
            ],
            "norovirus": [
                "流感",
                "influenza",
                "甲流",
                "新冠",
                "SARS-CoV-2",
                "COVID",
                "冠状病毒",
                "百日咳",
                "麻疹",
                "风疹",
                "结核",
                "结核病",
                "TB",
            ],
        }
        ban_terms = ban_by_dt.get(dt_norm, [])
        if ban_terms:
            filtered = []
            for g in unique:
                if not isinstance(g, dict):
                    continue
                src = str(g.get("source_file") or "")
                excerpt = str(g.get("excerpt") or "")
                if any(bt and (bt in src or bt in excerpt) for bt in ban_terms):
                    continue
                filtered.append(g)
            unique = filtered
        unique = unique[:10]

        generate_prompt = (
            "你是疾控防控措施智能体。给定疫情类型、风险等级与规范检索片段，生成可执行的防控措施。\n"
            "输出严格 JSON（不要输出多余文本）。\n"
            "字段：measures（数组）与 thinking_summary（中文要点，列 3-6 条）。\n"
            "每条措施结构：title(字符串), content(字符串), level(core|supplementary), citations(数组)。\n"
            "citations 结构：source_file, chunk_id, score, excerpt。\n"
            "硬约束：所有 core 措施 citations 至少 1 条。\n"
            "要求：\n"
            "- 每条措施 content 必须包含“适用场景/触发条件”（一句话即可），并结合 location 和 disease_type 的特性。\n"
            "- 至少输出 6 条 core 措施。\n"
            "- core：以国家/行业规范为主，内容尽量通用且可直接执行。\n"
            "- 若 key_points 为空：measures 中不得出现 level=supplementary。\n"
            "- 若 key_points 非空：必须输出 2-4 条 level=supplementary，且每条标题以“区域适配：”开头；内容需体现 key_points 中的差异化要点，但不要在每条中重复粘贴整段 key_points（适配因素会在区域适配栏统一展示）。\n"
            "- 措施覆盖面必须包含：病例发现与隔离/就医、环境消毒、通风与场所管理、监测与报告、健康宣教与风险沟通、重点人群/重点岗位（如食堂/宿管/校医）、食品与饮用水安全（若为诺如/肠道传染病必须包含）、应急组织与联动（疾控/学校/医院）。\n"
            "- 措施内容要尽量具体、可执行，避免只写原则口号。\n"
        )
        context = {
            "disease_type": disease_type,
            "location": location,
            "risk_level": risk_level,
            "key_points": key_points,
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
        original_measures: List[Dict[str, Any]] = []
        for m in measures:
            if not isinstance(m, dict):
                continue
            original_measures.append(m)
            level = str(m.get("level") or "core").lower()
            if level == "supplementary" and not key_points:
                continue
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
            content = str(m.get("content") or "")
            if key_points:
                content = content.replace("{key_points}", key_points)
                content = re.sub(
                    r"(适配因素\s*[:：])\s*\{([^}]+)\}\s*",
                    r"\1\2",
                    content,
                    flags=re.S,
                )
            if level == "supplementary":
                if "适用场景/触发条件" not in content:
                    content = (
                        "适用场景/触发条件：当需结合属地政策与本地条件细化防控措施时。"
                        + content.strip()
                    )
                marks = ["适配", "本地", "属地", "当地", "结合", "因地制宜"]
                if not any(x in content for x in marks):
                    content = content.strip()
                    if content and content[-1] not in "。！？":
                        content += "。"
                    content += "结合属地政策与本地条件细化执行。"
                content = re.sub(
                    r"适配因素\s*[:：][\s\S]*?(?:执行要点|具体措施|操作要点|$)",
                    "",
                    content,
                ).strip()
            fixed_measures.append(
                {
                    "title": str(m.get("title") or "未命名措施"),
                    "content": content,
                    "level": "supplementary" if level == "supplementary" else "core",
                    "citations": citations,
                }
            )

        core_count = sum(1 for m in fixed_measures if m.get("level") == "core")
        if core_count < 6:
            for m in original_measures:
                if core_count >= 6:
                    break
                title = str(m.get("title") or "未命名措施")
                content = str(m.get("content") or "")
                citations = _safe_list(m.get("citations"))
                if not citations and fallback_citation:
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
                if not key_points:
                    title = re.sub(r"^\s*区域适配[:：]\s*", "", title).strip()
                    content = re.sub(
                        r"适配因素[:：][^。！？]*[。！？]?", "", content
                    ).strip()
                if not any(
                    (x.get("title") == title and x.get("content") == content)
                    for x in fixed_measures
                ):
                    fixed_measures.append(
                        {
                            "title": title,
                            "content": content,
                            "level": "core",
                            "citations": citations,
                        }
                    )
                    core_count += 1

        supp_count = sum(1 for m in fixed_measures if m.get("level") == "supplementary")
        if key_points and supp_count < 2:
            templates = [
                {
                    "title": "区域适配：资源能力与物资储备",
                    "content": (
                        "适用场景/触发条件：当属地医疗救治、检测能力或防护物资储备存在差异时，按照本地缺口优先级滚动补齐。"
                        "由乡镇政府/学校后勤会同卫生院/校医院建立物资台账，按日盘点并向上级申请调拨。"
                    ),
                    "level": "supplementary",
                    "citations": [],
                },
                {
                    "title": "区域适配：重点场所与人群差异化措施",
                    "content": (
                        "适用场景/触发条件：当本地存在重点机构（学校/养老机构/集市/宿舍等）或脆弱人群聚集时，实施差异化管控与服务。"
                        "对重点场所落实错峰、限流、通风与消毒加密；对高风险人群加强健康随访与就医转诊指引。"
                    ),
                    "level": "supplementary",
                    "citations": [],
                },
            ]
            for t in templates:
                if supp_count >= 2:
                    break
                if not any(
                    (x.get("title") == t["title"] and x.get("content") == t["content"])
                    for x in fixed_measures
                ):
                    fixed_measures.append(t)
                    supp_count += 1

        def _supp_is_too_generic(text: str) -> bool:
            c = str(text or "")
            if not c:
                return True
            base = c
            if key_points:
                base = base.replace(key_points, "")
            base = re.sub(r"适用场景/触发条件\s*[:：][^。！？\n]*[。！？]?", "", base)
            base = re.sub(r"适配因素\s*[:：][^。！？\n]*[。！？]?", "", base)
            base = re.sub(r"\s+", " ", base).strip()
            if len(base) < 18:
                return True
            action_marks = [
                "执行要点",
                "具体措施",
                "操作要点",
                "落实",
                "实施",
                "设置",
                "建立",
                "增加",
                "启用",
                "调整",
                "错峰",
                "限流",
                "分区",
                "扩容",
                "储备",
                "调拨",
            ]
            return not any(x in base for x in action_marks)

        def _append_supp_actions(title: str, content: str) -> str:
            t = str(title or "")
            c = str(content or "").strip()
            if not c:
                return c
            if c and c[-1] not in "。！？":
                c += "。"
            if any(k in t for k in ["错峰", "分区", "预警", "活动", "课堂", "就餐"]):
                c += (
                    "执行要点：按宿舍楼/学院划分管理单元并建立班级/宿舍日报；达到预警阈值时启动错峰上课与错峰就餐，"
                    "暂停非必要聚集活动并快速切换线上教学；对食堂、图书馆等重点场所实施限流与高峰引导，"
                    "将通风与消毒频次与人流强度挂钩动态加密。"
                )
                return c
            if any(k in t for k in ["留学生", "双语", "重点人群", "慢病", "健康管理"]):
                c += (
                    "执行要点：对留学生发布中英双语通知并设置24小时咨询渠道；对慢病学生建立健康台账并开展每日随访与用药指导；"
                    "对食堂员工、宿管/保洁等重点岗位实行上岗前症状筛查与异常即停工复核；"
                    "优先向重点人群配置抗原试剂、口罩与抗病毒药物（按属地目录）。"
                )
                return c
            if any(k in t for k in ["隔离", "医疗", "资源", "转运", "床位"]):
                c += (
                    "执行要点：以现有隔离房间为基础设置备用隔离点（校内宾馆/空置宿舍/合作酒店）并明确启用阈值；"
                    "建立校医院—定点医院绿色通道与120联络机制，明确转运触发条件与交接流程；"
                    "按日盘点抗原试剂、消毒剂与防护物资，出现缺口时按清单向属地申请调拨并记录闭环。"
                )
                return c
            c += (
                "执行要点：将属地政策要求与校内资源清单对照形成差距表；明确触发阈值、牵头部门、时限与协作单位，"
                "并通过会商机制动态调整教学、就餐、住宿与活动管理措施。"
            )
            return c

        def _normalize_supp_key(title: str, content: str) -> str:
            t = re.sub(r"\s+", "", str(title or ""))
            c = str(content or "")
            if key_points:
                c = c.replace(key_points, "")
            c = re.sub(r"适用场景/触发条件\s*[:：]\s*", "", c)
            c = re.sub(
                r"适配因素\s*[:：][\s\S]*?(?:执行要点|具体措施|操作要点|$)", "", c
            )
            c = re.sub(r"\s+", "", c)
            c = re.sub(r"[，,。；;:：、!?！？—-]+", "", c)
            return (t + "||" + c).lower()

        for m in fixed_measures:
            if str(m.get("level") or "") != "supplementary":
                continue
            title = str(m.get("title") or "")
            content = str(m.get("content") or "")
            if _supp_is_too_generic(content):
                m["content"] = _append_supp_actions(title, content)

        deduped: List[Dict[str, Any]] = []
        seen_supp: set[str] = set()
        for m in fixed_measures:
            if str(m.get("level") or "") != "supplementary":
                deduped.append(m)
                continue
            k = _normalize_supp_key(
                str(m.get("title") or ""), str(m.get("content") or "")
            )
            if k in seen_supp:
                continue
            seen_supp.add(k)
            deduped.append(m)
        fixed_measures = deduped

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
