import json
import re
from typing import Any, Dict, List, Optional

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
        "other": "other",
    }
    known = set(mapping.values())
    if v in mapping:
        return mapping[v]
    if v in known:
        return v
    return "other"


def _disease_query_terms(disease_type: str) -> str:
    v = (disease_type or "").strip().lower()
    if v == "covid19":
        return "新冠 COVID-19 SARS-CoV-2 covid19 新型冠状病毒"
    if v == "influenza":
        return "流感 influenza 甲流"
    if v == "norovirus":
        return "诺如 norovirus 诺如病毒"
    if v == "measles_rubella":
        return "麻疹 风疹 measles rubella"
    if v == "pertussis":
        return "百日咳 pertussis"
    if v == "tuberculosis":
        return "结核病 肺结核 tuberculosis TB"
    if v == "dengue":
        return "登革热 dengue"
    if v == "hand_foot_mouth":
        return "手足口病 HFMD"
    if v == "varicella":
        return "水痘 varicella"
    if v == "mumps":
        return "流行性腮腺炎 mumps"
    if v == "hepatitis_a":
        return "甲型肝炎 甲肝 hepatitis A"
    if v == "food_poisoning":
        return "食物中毒 食源性"
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
    kb_only: bool = False

    @staticmethod
    def _strip_title_prefix(t: str) -> str:
        s = str(t or "").strip()
        s = re.sub(r"^(第)?[一二三四五六七八九十]+[、.]\s*", "", s)
        s = re.sub(r"^\d+[、.]\s*", "", s)
        return s

    @staticmethod
    def _sanitize_plan_title(title: str) -> str:
        s = str(title or "").strip()
        if not s:
            return ""
        s = re.sub(r"\s*[（(]\s*(修订版|优化版)\s*[）)]\s*$", "", s).strip()
        s = re.sub(r"\s*(修订版|优化版)\s*$", "", s).strip()
        return s

    @staticmethod
    def _strip_inline_outline_marks(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s)

        s = re.sub(r"[（(][一二三四五六七八九十]+[）)]", "", s)
        s = re.sub(r"(^|[。；;：:]\s*)[（(]\s*\d+\s*[）)]\s*", r"\1", s)

        s = re.sub(r"(^|[。；;：:]\s*)(第)?[一二三四五六七八九十]+[、.．]\s*", r"\1", s)
        s = re.sub(r"(^|[。；;：:]\s*)\d+[、.．]\s*", r"\1", s)
        s = re.sub(r"(^|[。；;：:]\s*)\d+[)）]\s*", r"\1", s)

        s = re.sub(r"^\s*(第\s*)?\d+\s*(章|节|条)\s*", "", s)
        s = re.sub(r"^\s*(第\s*)?[一二三四五六七八九十]+\s*(章|节|条)\s*", "", s)

        for _ in range(2):
            head = s.split("。", 1)[0].strip()
            if 0 < len(head) <= 28 and (
                ("（" in head and "）" in head)
                or ("(" in head and ")" in head)
                or any(
                    k in head
                    for k in ["定义", "处置", "原则", "要求", "流程", "时限", "阈值"]
                )
            ):
                if "：" in head or ":" in head:
                    break
                if not any(
                    k in head
                    for k in [
                        "应当",
                        "需要",
                        "必须",
                        "落实",
                        "开展",
                        "组织",
                        "指导",
                        "建立",
                    ]
                ):
                    tail = s.split("。", 1)
                    if len(tail) == 2 and tail[1].strip():
                        s = tail[1].strip()
                        continue
            break

        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^[，,。；;:：)\]】》】]+", "", s).strip()
        s = re.sub(r"：\s*[，,。；;:：、!?！？—-]+\s*", "：", s).strip()
        return s

    @staticmethod
    def _is_paragraph_detailed(p: Any) -> bool:
        return len(str(p or "").strip()) >= 20

    @classmethod
    def _is_section_shallow(cls, sec: Dict[str, Any]) -> bool:
        paras = sec.get("paragraphs")
        if not isinstance(paras, list) or len(paras) < 3:
            return True
        return any(not cls._is_paragraph_detailed(p) for p in paras[:3])

    @staticmethod
    def _kw_score(text: str, keywords: List[str]) -> int:
        s = (text or "").lower()
        score = 0
        for kw in keywords:
            kw2 = (kw or "").strip().lower()
            if not kw2:
                continue
            if kw2 in s:
                score += 2
        return score

    @staticmethod
    def _split_terms(text: str) -> List[str]:
        s = str(text or "").strip()
        if not s:
            return []
        parts = re.split(r"[\s,/，、]+", s)
        return [p.strip() for p in parts if p and p.strip()]

    @classmethod
    def _pick_evidence(
        cls,
        guidelines: List[Dict[str, Any]],
        keywords: List[str],
        must_terms: Optional[List[str]] = None,
        prefer_terms: Optional[List[str]] = None,
        slot_name: Optional[str] = None,
        location: Optional[str] = None,
        top_n: int = 2,
        exclude_keys: Optional[set] = None,
        ban_terms: Optional[List[str]] = None,
        min_score: int = 2,
    ) -> List[Dict[str, Any]]:
        scored = []
        exclude_keys = exclude_keys or set()
        ban_terms = ban_terms or []
        must_terms = must_terms or []
        prefer_terms = prefer_terms or []
        slot_name = str(slot_name or "").strip()
        bg_terms = [
            "发病机制",
            "机制尚不明确",
            "潜伏期",
            "病程",
            "临床表现",
            "诊断",
            "治疗",
            "pH",
            "乙醇",
            "75%",
            "60℃",
            "冷藏",
            "冷冻",
        ]
        prefer_no_bg_slots = {
            "指挥体系与职责",
            "联动机制",
            "值班与报告链路",
            "预警阈值",
            "报告流程与时限",
            "风险沟通与信息发布",
            "监督检查与记录",
            "解释与修订",
            "实施与发布",
            "附表与联系方式",
        }
        macro_terms = [
            "党中央",
            "国务院",
            "中央",
            "全国",
            "国家安全",
            "中央和国家机关",
            "国家机关",
            "应急管理部",
            "立体化监测预警网络",
            "大数据支撑",
            "智慧应急",
            "数字化能力建设",
            "视频会商",
            "辅助决策",
            "资源调用",
            "预案管理",
            "国家相关应急指挥机构",
            "牵头编制部门",
        ]
        macro_sensitive_slots = {"目的与适用范围", "指挥体系与职责", "联动机制"}
        loc = str(location or "").strip()
        is_local = any(k in loc for k in ["乡", "镇", "村", "社区", "街道"])
        for g in guidelines or []:
            if not isinstance(g, dict):
                continue
            excerpt = str(g.get("excerpt") or "")
            src = str(g.get("source_file") or "")
            if not excerpt.strip() or not src.strip():
                continue
            if any(bt and bt in excerpt for bt in ban_terms):
                continue
            key = (src, str(g.get("chunk_id") or ""))
            if key in exclude_keys:
                continue
            if (
                is_local
                and slot_name in macro_sensitive_slots
                and any(mt and mt in excerpt for mt in macro_terms)
            ):
                continue
            if (
                slot_name in prefer_no_bg_slots
                and any(bt and bt in excerpt for bt in bg_terms)
                and (
                    not must_terms or not any(mt and mt in excerpt for mt in must_terms)
                )
            ):
                continue
            s = cls._kw_score(excerpt, keywords) + cls._kw_score(src, keywords)
            for mt in must_terms:
                mt2 = (mt or "").strip()
                if mt2 and mt2 in excerpt:
                    s += 2
            for pt in prefer_terms:
                pt2 = (pt or "").strip()
                if pt2 and pt2 in excerpt:
                    s += 1
            if s < int(min_score):
                continue
            scored.append((s, g))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        seen = set()
        for _, g in scored:
            key = (str(g.get("source_file")), str(g.get("excerpt")))
            if key in seen:
                continue
            seen.add(key)
            out.append(g)
            if len(out) >= top_n:
                break
        return out

    @staticmethod
    def _slot_semantic_ok(slot_name: str, text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return False
        name = str(slot_name or "").strip()
        bad_bg = ["IgG", "抗体", "滴度", "灭活", "紫外线", "56℃", "敏感性", "特异性"]
        political = ["习近平", "新时代", "中国特色社会主义思想", "两个维护", "两个确立"]
        school_monitor_terms = [
            "学生",
            "师生",
            "教职工",
            "校医院",
            "卫生所",
            "晨检",
            "午检",
            "晨午检",
            "因病缺勤",
            "缺勤",
            "缺课",
            "就诊",
            "发热",
            "症状",
        ]
        if name == "目的与适用范围":
            if any(k in s for k in political):
                return False
            if any(
                k in s
                for k in [
                    "立体化监测预警网络",
                    "大数据支撑",
                    "智慧应急",
                    "视频会商",
                    "辅助决策",
                    "资源调用",
                    "预案管理",
                    "中央和国家机关",
                    "国务院",
                    "党中央",
                ]
            ):
                return False
            has_scope = ("本预案" in s) or ("本方案" in s)
            has_apply = any(k in s for k in ["适用", "适用于", "用于", "范围"])
            has_scene = any(k in s for k in ["学校", "校园", "师生", "某大学", "本校"])
            return has_scope and has_apply and has_scene
        if name == "预警阈值":
            diag_terms = [
                "临床表现",
                "流行病学史",
                "病原学",
                "抗原检测",
                "核酸检测",
                "阳性",
                "确诊病例定义",
                "诊断标准",
            ]
            if any(k in s for k in diag_terms):
                return False
            has_trigger = any(
                k in s for k in ["达到", "超过", "高于", "触发", "启动预警", "≥", ">="]
            )
            has_num = bool(re.search(r"\d+(\.\d+)?", s)) or ("≥" in s) or (">=" in s)
            has_case = ("例" in s) or ("病例" in s)
            has_window = any(k in s for k in ["日", "天", "/日", "每日报告", "连续"])
            return has_trigger and has_num and has_case and has_window
        if name == "报告流程与时限":
            has_report = ("报告" in s) or ("上报" in s) or ("直报" in s)
            has_time = ("小时" in s) or ("当日" in s) or ("内" in s) or ("及时" in s)
            return has_report and has_time
        if name == "编制依据":
            if any(
                k in s
                for k in [
                    "不明原因肺炎",
                    "危险化学品",
                    "输电",
                    "抢修",
                    "灭活",
                    "紫外线",
                    "乙醇",
                    "75%",
                    "氯仿",
                    "过氧乙酸",
                    "违法行为",
                    "调查处理",
                    "技术标准",
                    "协助卫生行政部门",
                ]
            ):
                return False
            has_basis = ("依据" in s) or ("按照" in s) or ("参照" in s)
            has_ref = any(
                k in s for k in ["法", "条例", "预案", "方案", "指南", "规范", "通知"]
            )
            return has_basis and has_ref
        if name == "监测对象与指标":
            core_school_terms = [
                "学生",
                "师生",
                "教职工",
                "晨检",
                "午检",
                "晨午检",
                "因病缺勤",
                "缺勤",
                "缺课",
                "就诊",
                "发热",
                "症状",
            ]
            if ("污水" in s or "排污口" in s) and not any(
                k in s for k in core_school_terms
            ):
                return False
            has_school = any(k in s for k in school_monitor_terms)
            if has_school:
                return True
            return any(k in s for k in ["监测", "检测", "哨点", "采样", "监测点"])
        if name == "指挥体系与职责":
            has_cmd = any(k in s for k in ["指挥部", "领导小组", "工作组", "专班"])
            has_role = any(
                k in s for k in ["职责", "负责", "牵头", "分工", "统筹", "协调"]
            )
            if any(k in s for k in bad_bg):
                return False
            return has_cmd and has_role
        if name == "联动机制":
            has_joint = any(
                k in s for k in ["联动", "协同", "信息共享", "会商", "联防联控"]
            )
            if any(k in s for k in bad_bg):
                return False
            has_partners = any(
                k in s
                for k in [
                    "属地",
                    "疾控",
                    "定点医院",
                    "教育主管部门",
                    "校医院",
                    "卫生健康部门",
                    "社区卫生服务中心",
                ]
            )
            if (
                any(
                    k in s
                    for k in [
                        "相邻地区",
                        "区域性",
                        "流域性",
                        "关联性强",
                        "本区域应急管理",
                        "共同做好",
                    ]
                )
                and not has_partners
            ):
                return False
            return has_joint and has_partners
        if name == "值班与报告链路":
            has_duty = ("值班" in s) or ("24小时" in s) or ("每日" in s)
            has_report = (
                ("报告" in s) or ("上报" in s) or ("通报" in s) or ("直报" in s)
            )
            has_time = ("小时" in s) or ("当日" in s) or ("内" in s) or ("及时" in s)
            return has_duty and has_report and has_time
        if name == "工作原则":
            bad = [
                "军队",
                "突击力量",
                "财政事权",
                "支出责任",
                "危险化学品",
                "封锁",
                "封闭管理",
                "全员核酸",
                "分区管控",
                "电力",
                "输电",
                "抢修",
                "传染源管理",
                "隔离措施",
                "分级分类收治",
                "抗原",
                "核酸",
                "病原学",
                "流行病学特征",
                "SARS-",
                "食品安全",
                "报送我部",
                "体育卫生与艺术教育司",
                "辖区",
                "社区",
                "基层医疗卫生机构",
                "网底",
                "居（村）",
                "居委",
                "卫生院",
                "生活垃圾",
                "65岁",
                "老年",
            ]
            if any(k in s for k in bad):
                return False
            if any(k in s for k in political):
                return False
            good = [
                "属地",
                "联防联控",
                "依法",
                "科学",
                "分级",
                "协同",
                "预防为主",
                "早发现",
                "早报告",
                "早处置",
                "精准",
                "快速",
                "公开透明",
            ]
            return any(k in s for k in good)
        if name == "响应终止条件":
            if any(k in s for k in bad_bg):
                return False
            has_last = any(k in s for k in ["最后", "末例", "最后1例", "最后一例"])
            has_no_new = any(
                k in s for k in ["无新增", "无新发", "连续", "最长潜伏期", "潜伏期"]
            )
            has_eval = any(k in s for k in ["评估", "研判", "终止", "解除"])
            return has_last and has_no_new and has_eval
        if name == "评估与复盘":
            return any(k in s for k in ["评估", "复盘", "总结", "修订", "改进"])
        if name == "恢复与关怀":
            if any(k in s for k in ["IgG", "抗体", "滴度"]):
                return False
            return any(
                k in s for k in ["心理", "疏导", "关怀", "复课", "复工", "恢复", "慰问"]
            )
        if name == "物资与经费保障":
            has_any = any(
                k in s
                for k in [
                    "物资",
                    "储备",
                    "采购",
                    "调拨",
                    "经费",
                    "预算",
                    "台账",
                    "缺口",
                ]
            )
            has_core = any(
                k in s for k in ["储备", "采购", "调拨", "经费", "预算", "缺口"]
            )
            return has_any and has_core
        if name == "队伍与技术保障":
            has_team = any(
                k in s for k in ["队伍", "人员", "培训", "演练", "保障", "能力", "技术"]
            )
            if any(
                k in s
                for k in [
                    "口岸",
                    "入境",
                    "全基因组测序",
                    "基因组测序",
                    "报送中国疾控中心",
                ]
            ):
                return False
            if any(
                k in s
                for k in [
                    "中国疾病预防控制中心",
                    "省级疾病预防控制中心",
                    "县级以上疾病预防控制机构",
                ]
            ) and not any(k in s for k in ["学校", "校医院", "卫生所", "后勤", "学生处"]):
                return False
            if (
                any(
                    k in s
                    for k in ["敏感性", "特异性", "区分病毒", "区分病毒类型", "亚型"]
                )
                and not has_team
            ):
                return False
            return has_team
        if name == "监督检查与记录":
            if any(k in s for k in ["卫生监督机构", "执法", "稽查"]):
                return False
            return any(
                k in s for k in ["监督", "检查", "记录", "台账", "督导", "整改", "考核"]
            )
        if name == "启动条件与分级":
            if any(k in s for k in bad_bg):
                return False
            if any(
                k in s
                for k in [
                    "核酸检测点",
                    "不再实行隔离",
                    "不再判定密切接触者",
                    "不再划定高低风险区",
                    "分级分类收治",
                    "65岁",
                    "老年",
                    "基层医疗卫生机构",
                    "社区（村）",
                    "居（村）民委员会",
                ]
            ):
                return False
            has_start = any(k in s for k in ["启动", "响应", "进入", "启动响应"])
            has_level = any(
                k in s
                for k in [
                    "分级",
                    "Ⅰ",
                    "Ⅱ",
                    "Ⅲ",
                    "I级",
                    "II级",
                    "III级",
                    "一级",
                    "二级",
                    "三级",
                ]
            )
            has_condition = any(
                k in s
                for k in [
                    "达到",
                    "超过",
                    "出现",
                    "当",
                    "如",
                    "满足",
                    "阈值",
                    "标准",
                    "触发",
                    "≥",
                    ">=",
                ]
            ) and (bool(re.search(r"\d+(\.\d+)?", s)) or ("≥" in s) or (">=" in s))
            return has_start and has_level and has_condition
        if name == "处置流程":
            bad = [
                "我健康监测",
                "第一责任人",
                "倡导公众",
                "提高健康素养",
                "全方位、多渠道开展",
                "勤洗手",
                "公筷",
                "合理膳食",
            ]
            if any(k in s for k in bad):
                return False
            actions = [
                "发现",
                "检测",
                "隔离",
                "报告",
                "流调",
                "密接",
                "转诊",
                "消毒",
            ]
            action_hits = sum(1 for k in actions if k in s)
            if action_hits >= 3:
                return True
            if action_hits >= 2 and any(
                k in s for k in ["立即", "同时", "并", "随后", "然后"]
            ):
                return True
            return False
        if name == "风险沟通与信息发布":
            return any(
                k in s for k in ["发布", "通报", "沟通", "谣言", "舆情", "公告", "热线"]
            )
        if name == "解释与修订":
            return any(k in s for k in ["解释", "修订", "修正", "两年", "定期"])
        if name == "实施与发布":
            if any(
                k in s for k in ["隔离措施", "核酸", "抗原", "密切接触者", "高低风险区"]
            ):
                return False
            if any(k in s for k in bad_bg) or any(
                k in s for k in ["病原学", "流行病学特征", "SARS-CoV-2", "β属冠状病毒"]
            ):
                return False
            return any(
                k in s
                for k in [
                    "自发布之日起",
                    "发布之日起",
                    "印发之日起",
                    "自印发之日起",
                    "生效",
                    "实施",
                ]
            )
        if name == "附表与联系方式":
            if any(
                k in s
                for k in [
                    "教育部",
                    "moe.edu.cn",
                    "报送我部",
                    "体育卫生与艺术教育司",
                    "食品安全",
                ]
            ):
                return False
            has_contact = any(
                k in s for k in ["联系方式", "电话", "邮箱", "附表", "联系人"]
            )
            has_local_owner = any(
                k in s
                for k in [
                    "校医院",
                    "卫生所",
                    "学生处",
                    "学生工作",
                    "后勤",
                    "疾控",
                    "定点医院",
                    "120",
                ]
            )
            return has_contact and has_local_owner
        return True

    @staticmethod
    def _parse_slot_paragraph(p: str) -> Optional[tuple[str, str]]:
        s = str(p or "").strip()
        if not s:
            return None
        s = re.sub(r"^\s*[（(][^）)]*[）)]\s*", "", s).strip()
        if "：" in s:
            left, right = s.split("：", 1)
        elif ":" in s:
            left, right = s.split(":", 1)
        else:
            return None
        return left.strip(), right.strip()

    @classmethod
    def _section_slots_semantic_ok(cls, title: str, paragraphs: Any) -> bool:
        t = cls._strip_title_prefix(str(title or "")).strip()
        paras = paragraphs if isinstance(paragraphs, list) else []
        required = {
            "后期处置": ["响应终止条件", "评估与复盘", "恢复与关怀"],
            "保障措施": ["物资与经费保障", "队伍与技术保障", "监督检查与记录"],
            "总则": ["目的与适用范围", "编制依据", "工作原则"],
        }
        slots = required.get(t)
        if not slots:
            return True
        found: Dict[str, str] = {}
        for p in paras:
            parsed = cls._parse_slot_paragraph(str(p or ""))
            if not parsed:
                continue
            k, v = parsed
            for slot in slots:
                if k == slot and v:
                    found[slot] = v
        for slot in slots:
            v = found.get(slot, "")
            if not v:
                return False
            if not cls._slot_semantic_ok(slot, v):
                return False
        return True

    @staticmethod
    def _default_slot_terms(slot_name: str) -> tuple[list[str], list[str]]:
        name = str(slot_name or "").strip()
        must_map: Dict[str, List[str]] = {
            "目的与适用范围": ["目的", "适用", "范围"],
            "编制依据": ["依据", "按照", "参照"],
            "工作原则": ["属地", "联防联控", "依法", "科学", "分级", "协同"],
            "指挥体系与职责": ["指挥", "工作组", "职责", "负责"],
            "值班与报告链路": ["值班", "报告", "24小时", "电话"],
            "监测对象与指标": ["学生", "教职工", "晨午检", "因病缺勤", "校医院"],
            "响应终止条件": ["终止", "解除", "评估", "无新增", "连续", "潜伏期"],
            "评估与复盘": ["评估", "复盘", "总结", "修订", "改进"],
            "恢复与关怀": ["恢复", "复课", "复工", "心理", "关怀"],
            "物资与经费保障": ["物资", "储备", "采购", "经费", "调拨"],
            "队伍与技术保障": ["队伍", "培训", "演练", "技术", "能力"],
            "监督检查与记录": ["监督", "检查", "记录", "台账", "整改"],
            "启动条件与分级": ["启动", "条件", "分级", "响应"],
            "处置流程": ["处置", "流程", "隔离", "转诊", "消毒"],
            "风险沟通与信息发布": ["发布", "通报", "沟通", "舆情"],
            "解释与修订": ["解释", "修订", "两年", "定期"],
            "实施与发布": ["发布", "实施", "生效"],
            "附表与联系方式": ["联系方式", "联系人", "电话"],
        }
        ban_map: Dict[str, List[str]] = {
            "目的与适用范围": ["危险化学品", "封锁", "电力"],
            "编制依据": [
                "不明原因肺炎",
                "危险化学品",
                "电力",
                "紫外线",
                "灭活",
                "75%",
                "乙醇",
                "氯仿",
                "过氧乙酸",
            ],
            "工作原则": [
                "军队",
                "突击力量",
                "财政事权",
                "支出责任",
                "电力",
                "食品安全",
                "体育卫生与艺术教育司",
                "报送我部",
                "65岁",
                "老年",
                "辖区",
                "社区",
                "基层医疗卫生机构",
                "网底",
                "居（村）",
                "居委",
                "卫生院",
                "生活垃圾",
            ],
            "启动条件与分级": [
                "紫外线",
                "灭活",
                "56℃",
                "敏感性",
                "特异性",
                "乙醇",
                "氯仿",
                "75%",
            ],
            "监测对象与指标": ["污水", "排污口"],
            "恢复与关怀": ["IgG", "抗体", "滴度", "电力", "输电"],
            "实施与发布": ["危险化学品", "封锁"],
        }
        return must_map.get(name, []), ban_map.get(name, [])

    async def _llm_rewrite_slot_value(
        self,
        *,
        section_title: str,
        slot_name: str,
        current_value: str,
        disease_type: str,
        location: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        dt_norm = str(disease_type or "").strip().lower()
        disease_terms: Dict[str, List[str]] = {
            "covid19": ["新型冠状病毒感染", "新冠", "COVID-19"],
            "influenza": ["流感"],
            "norovirus": ["诺如病毒"],
            "measles_rubella": ["麻疹", "风疹"],
            "pertussis": ["百日咳"],
            "tuberculosis": ["结核", "肺结核", "结核病"],
            "dengue": ["登革热"],
            "hand_foot_mouth": ["手足口病", "手足口"],
            "varicella": ["水痘"],
            "mumps": ["腮腺炎", "流行性腮腺炎"],
            "hepatitis_a": ["甲肝", "甲型肝炎"],
            "food_poisoning": ["食物中毒"],
        }
        allow_terms = disease_terms.get(dt_norm, [])
        forbid_terms: List[str] = []
        for k, terms in disease_terms.items():
            if k == dt_norm:
                continue
            forbid_terms.extend(terms)

        must_include = []
        if allow_terms and slot_name in {"目的与适用范围", "编制依据"}:
            must_include = allow_terms[:2]

        lines = [
            "你是公共卫生应急预案语义校验与修复助手。",
            "任务：判断 current_value 是否符合 slot_name 的语义；若不符合，重写为可直接放入预案的表述。",
            "约束：",
            "- 输出严格 JSON，仅包含 ok(布尔), revised(字符串), reason(字符串)。",
            "- revised 仅输出“冒号后面的值”，不要包含 slot_name 与冒号。",
            "- revised 必须以“（AI校验修复）”开头。",
            "- revised 不得出现与 slot_name 无关的领域（如电力/危险化学品/军队等）。",
            f"- revised 不得出现其他疾病：{('、'.join(forbid_terms) if forbid_terms else '（无）')}。",
        ]
        if slot_name == "附表与联系方式":
            lines.extend(
                [
                    "- revised 不得包含真实个人姓名、个人手机号/固话、个人邮箱等个人信息。",
                    "- revised 仅给出“部门/岗位+联系电话/邮箱（待填）”的占位内容，并包含属地疾控/定点医院/120等联动对象。",
                    "- revised 不得出现教育部/moe.edu.cn/体育卫生与艺术教育司等外部单位联系人信息。",
                ]
            )
        if must_include:
            lines.append(
                f"- revised 必须至少包含以下疾病关键词之一：{('、'.join(must_include))}。"
            )
        lines.extend(
            [
                "- revised 不要使用小标题序号（如（一）/1./2. 等）。",
                "- revised 内容要具体可执行，尽量包含触发条件/责任主体/时限（能写则写）。",
            ]
        )
        sys_prompt = "\n".join(lines) + "\n"
        payload = {
            "section_title": section_title,
            "slot_name": slot_name,
            "disease_type": disease_type,
            "location": location,
            "current_value": current_value,
            "evidence": [
                {
                    "source_file": str(g.get("source_file") or ""),
                    "chunk_id": g.get("chunk_id"),
                    "excerpt": str(g.get("excerpt") or ""),
                }
                for g in (evidence or [])[:3]
                if isinstance(g, dict)
            ],
        }
        raw = await self.llm.ask(
            messages=[Message.user_message(json.dumps(payload, ensure_ascii=False))],
            system_msgs=[Message.system_message(sys_prompt)],
            stream=False,
            temperature=0.0,
        )
        obj = _extract_json(raw)
        ok = bool(obj.get("ok")) if isinstance(obj.get("ok"), bool) else False
        revised = str(obj.get("revised") or "").strip()
        reason = str(obj.get("reason") or "").strip()
        if not revised:
            ok = True
        revised = re.sub(r"^[，,。；;:：、!?！？—-]+\s*", "", revised).strip()
        revised = re.sub(r"^（AI校验修复）\s*", "（AI校验修复）", revised).strip()
        if revised:
            revised2 = revised.replace("（AI校验修复）", "")
            if any(t and t in revised2 for t in forbid_terms):
                return {
                    "ok": False,
                    "revised": "",
                    "reason": "rewrite_contains_other_disease",
                }
            if slot_name == "附表与联系方式" and any(
                k in revised2
                for k in [
                    "教育部",
                    "moe.edu.cn",
                    "体育卫生与艺术教育司",
                    "报送我部",
                    "食品安全",
                ]
            ):
                return {
                    "ok": False,
                    "revised": "",
                    "reason": "rewrite_contains_external_contact",
                }
        return {"ok": ok, "revised": revised, "reason": reason}

    async def _llm_judge_slot_value(
        self,
        *,
        section_title: str,
        slot_name: str,
        current_value: str,
        disease_type: str,
        location: str,
    ) -> Dict[str, Any]:
        lines = [
            "你是公共卫生应急预案语义判定助手。",
            "任务：判断 current_value 是否适合作为预案中 slot_name 的正文内容（结合 disease_type 与 location 场景）。",
            "输出严格 JSON，仅包含 ok(布尔), reason(字符串)。",
            "判定规则：",
            "- ok=true：内容与 slot_name 语义一致，且可直接放入预案正文（允许存在轻微格式问题，如编号/小标题/多余空格）。",
            "- ok=false：内容明显跑题/串段/主体与场景不一致/夹带不应出现的技术条款或无关章节内容。",
            "- 不要尝试改写正文，不要输出 revised。",
        ]
        sys_prompt = "\n".join(lines) + "\n"
        payload = {
            "section_title": section_title,
            "slot_name": slot_name,
            "disease_type": disease_type,
            "location": location,
            "current_value": current_value,
        }
        raw = await self.llm.ask(
            messages=[Message.user_message(json.dumps(payload, ensure_ascii=False))],
            system_msgs=[Message.system_message(sys_prompt)],
            stream=False,
            temperature=0.0,
        )
        obj = _extract_json(raw)
        ok = bool(obj.get("ok")) if isinstance(obj.get("ok"), bool) else False
        reason = str(obj.get("reason") or "").strip()
        return {"ok": ok, "reason": reason}

    async def _semantic_repair_sections(
        self,
        *,
        sections: List[Dict[str, Any]],
        disease_type: str,
        location: str,
        guidelines: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        repairs: List[Dict[str, Any]] = []
        out: List[Dict[str, Any]] = []
        dt_norm = str(disease_type or "").strip().lower()
        disease_terms: Dict[str, List[str]] = {
            "covid19": ["新冠", "新型冠状病毒", "COVID", "covid"],
            "influenza": ["流感", "influenza"],
            "norovirus": ["诺如", "norovirus"],
            "measles_rubella": ["麻疹", "风疹"],
            "pertussis": ["百日咳"],
            "tuberculosis": ["结核", "结核病", "TB", "tb"],
        }
        allow_terms = disease_terms.get(dt_norm, [])

        def _mentions_other_disease(text: str) -> bool:
            s2 = str(text or "")
            if allow_terms and any(t and t in s2 for t in allow_terms):
                return False
            for k, terms in disease_terms.items():
                if k == dt_norm:
                    continue
                if any(t and t in s2 for t in terms):
                    return True
            return False

        def _missing_allowed_disease(text: str) -> bool:
            s2 = str(text or "")
            if not allow_terms:
                return False
            return not any(t and t in s2 for t in allow_terms)

        macro_terms = [
            "党中央",
            "国务院",
            "中央",
            "全国",
            "国家安全",
            "中央和国家机关",
            "国家机关",
            "应急管理部",
        ]
        political_terms = [
            "习近平",
            "新时代",
            "中国特色社会主义思想",
            "两个维护",
            "两个确立",
        ]
        is_school = any(
            k in str(location or "")
            for k in ["大学", "学院", "学校", "中学", "小学", "幼儿园", "托幼"]
        )
        loc_text = str(location or "").strip()
        school_place_bad = [
            "乡镇",
            "村",
            "村委",
            "村卫生室",
            "乡镇卫生院",
            "卫生院",
            "街道办",
            "街道办事处",
            "街道社区",
            "社区卫生服务中心",
            "社区居委",
        ]

        def _fallback_slot(slot_name: str) -> str:
            loc2 = str(location or "").strip() or "本地区"
            if dt_norm == "covid19":
                dname = "新型冠状病毒感染"
            elif dt_norm == "influenza":
                dname = "流感"
            elif dt_norm == "norovirus":
                dname = "诺如病毒感染"
            else:
                dname = str(disease_type or "相关传染病").strip() or "相关传染病"
            if slot_name == "目的与适用范围":
                return (
                    f"（AI校验修复）本预案用于指导{loc2}{dname}疫情应急处置工作，"
                    "明确监测预警、信息报告、应急响应、医疗救治与物资保障等要求，"
                    "以最大限度降低疫情对公众健康与社会运行的影响。"
                )
            if slot_name == "编制依据":
                return (
                    f"（AI校验修复）依据《中华人民共和国传染病防治法》等法律法规，"
                    f"参照国家及省市关于{dname}防控的最新方案/指南/应急预案，结合{loc2}实际制定。"
                )
            if slot_name == "工作原则":
                if is_school:
                    return (
                        "（AI校验修复）坚持预防为主、关口前移；坚持属地管理、校地协同、联防联控；"
                        "坚持早发现、早报告、早处置，科学精准分级响应；坚持信息公开透明与依法依规处置；"
                        "坚持重点人群与重点场所（宿舍/食堂/教室）分类管理。"
                    )
                return (
                    "（AI校验修复）坚持预防为主、关口前移；坚持属地管理、联防联控；"
                    "坚持早发现、早报告、早处置，科学精准分级响应；坚持信息公开透明与依法依规处置；"
                    "坚持重点人群与重点场所分类管理。"
                )
            if slot_name == "指挥体系与职责":
                if is_school:
                    return (
                        f"（AI校验修复）成立{loc2}{dname}疫情防控领导小组，由校党委书记/校长任组长，"
                        "统一指挥协调；下设综合协调、监测预警与报告、流调处置与密接管理、医疗救治与转运、"
                        "后勤保障与消毒、宣传与舆情、教学与学生管理等工作组，明确牵头部门、联络员与岗位职责，"
                        "实行清单化分工与每日会商机制。"
                    )
                return (
                    f"（AI校验修复）成立{loc2}{dname}疫情应急指挥部（或联防联控机制），"
                    "由主要负责同志担任总指挥，统一组织协调；下设综合协调、监测预警、流调处置、"
                    "医疗救治、物资保障、宣传与舆情等工作组，明确牵头单位与岗位职责，实行清单化分工落实。"
                )
            if slot_name == "联动机制":
                if is_school:
                    return (
                        f"（AI校验修复）建立与属地疾控中心、定点医院、教育主管部门的信息共享与联动机制，"
                        "明确会商频次、转运绿色通道、检测送检与结果反馈、物资调拨与应急支援流程；"
                        "发生聚集性疫情时按规定启动校地联合处置与现场指导。"
                    )
                return (
                    f"（AI校验修复）建立与县级疾控机构、定点医院的信息共享和应急联动机制，"
                    "强化会商研判、资源调配、转诊救治和跨区域协同处置，必要时按程序请求上级支援。"
                )
            if slot_name == "值班与报告链路":
                if is_school:
                    return (
                        f"（AI校验修复）实行24小时值班与信息报送制度。校医院/卫生所为首报责任部门，"
                        "发现聚集性疫情、重症或异常升高趋势时，立即向学校领导小组报告并电话报告属地疾控；"
                        "按规定时限开展网络直报（首报/续报/终报），每日汇总缺勤与就诊数据并在校内通报处置进展。"
                    )
                return (
                    f"（AI校验修复）实行24小时值班和疫情信息“首报—续报—终报”制度，"
                    "发现聚集性疫情、重症/死亡或异常升高趋势时，立即电话报告县级疾控和卫健部门，"
                    "并按规定时限通过网络直报系统报送；每日汇总并向指挥部通报监测与处置进展。"
                )
            if slot_name == "监测对象与指标":
                if is_school:
                    return (
                        "（AI校验修复）监测对象包括在校学生、教职工及校内从业人员（食堂/宿管/保洁等）。"
                        "指标包括晨午检发热/咳嗽等呼吸道症状人数、因病缺勤缺课人数与原因、校医院就诊与抗原检测阳性数、"
                        "宿舍/班级/学院聚集性病例数等，实行每日汇总、分层上报与趋势研判。"
                    )
                return (
                    "（AI校验修复）监测对象包括重点场所人员与重点岗位从业者。指标包括症状监测、因病缺勤、就诊与检测阳性、"
                    "聚集性事件等，实行每日汇总、分层报告与趋势研判。"
                )
            if slot_name == "预警阈值":
                return (
                    "（AI校验修复）当同一班级/宿舍楼24小时内出现3例及以上确诊病例，或全校当日新增发热伴呼吸道症状病例达到10例及以上，"
                    "或出现聚集性暴发趋势（3天内同一学院累计≥5例）时，启动预警并与属地疾控会商，视情采取停课/线上教学、"
                    "限制聚集活动等措施。"
                )
            if slot_name == "报告流程与时限":
                return (
                    "（AI校验修复）发现疑似聚集性疫情后，校医院/卫生所应立即核实并在2小时内电话报告属地疾控中心；"
                    "符合报告条件的按规定在24小时内完成网络直报，并根据病例变化及时订正续报；"
                    "学校同步向教育主管部门报告并启动校内信息通报与处置记录。"
                )
            if slot_name == "启动条件与分级":
                return (
                    "（AI校验修复）根据病例聚集程度分级启动响应：Ⅰ级（班级/宿舍聚集）启动班级/楼层处置；"
                    "Ⅱ级（学院/宿舍楼暴发）启动跨部门联合处置与错峰/停课；Ⅲ级（全校暴发）启动全校应急响应与线上教学。"
                    "满足预警阈值或出现重症/异常增幅时立即升级响应。"
                )
            if slot_name == "处置流程":
                return (
                    "（AI校验修复）实施“发现—检测—隔离—报告—流调—密接管理—环境消毒—健康宣教—复评”闭环处置："
                    "校医院预检分诊并开展抗原/核酸检测；阳性或疑似病例单间隔离并按绿色通道转诊；"
                    "对密接实行每日健康监测与必要时检测；对教室/宿舍/食堂开展重点消毒与通风；"
                    "每日会商评估并动态调整停课与活动管理措施。"
                )
            if slot_name == "风险沟通与信息发布":
                return (
                    "（AI校验修复）建立统一口径的信息发布机制，由学校宣传部门会同校医院与属地疾控发布权威信息，"
                    "定期通报疫情态势与防护要点；对师生开展风险沟通与答疑，及时澄清谣言，避免恐慌；"
                    "涉个人信息严格脱敏，做到公开透明与依法合规。"
                )
            if slot_name == "响应终止条件":
                return (
                    "（AI校验修复）当末例确诊病例治愈或解除隔离后，经过最长潜伏期无新增病例，"
                    "聚集性疫情处置完成且环境终末消毒到位，经属地疾控评估同意后，"
                    "由学校领导小组宣布终止应急响应并逐步恢复正常教学秩序。"
                )
            if slot_name == "评估与复盘":
                return (
                    "（AI校验修复）由学校领导小组牵头开展复盘评估，梳理病例发现、报告时效、隔离转运、"
                    "密接管理、消毒通风、物资保障与舆情沟通等环节问题，形成书面总结与整改清单，"
                    "按程序修订预案并组织演练验证。"
                )
            if slot_name == "恢复与关怀":
                return (
                    "（AI校验修复）疫情缓解后分批恢复线下教学与活动，持续开展两周健康随访；"
                    "对患病师生提供必要的学习支持与心理关怀，必要时由心理健康中心开展疏导；"
                    "对隔离与封控区域进行环境恢复与卫生整治，恢复常态化防控。"
                )
            if slot_name == "物资与经费保障":
                return (
                    "（AI校验修复）建立口罩、消毒剂、检测耗材、体温监测设备等物资储备与台账管理制度，"
                    "明确采购、调拨与缺口补齐流程；设立专项经费并明确审批与报销路径，"
                    "确保预警触发后可快速启动应急采购与配送。"
                )
            if slot_name == "队伍与技术保障":
                return (
                    "（AI校验修复）组建由校医院、后勤、学生工作、保卫与宣传等部门组成的应急队伍，"
                    "定期开展监测报告、隔离转运、消毒与个人防护等培训演练；"
                    "与属地疾控建立技术指导与检测支持机制，确保快速处置与规范操作。"
                )
            if slot_name == "监督检查与记录":
                return (
                    "（AI校验修复）建立监督检查与过程记录制度，对晨午检、缺勤追踪、病例隔离、"
                    "密接管理、消毒通风、物资发放与废弃物处置等关键环节开展督导检查；"
                    "形成台账与问题整改闭环，重要事项留痕备查。"
                )
            if slot_name == "解释与修订":
                return (
                    "（AI校验修复）本预案由学校疫情防控领导小组负责解释，并根据政策更新、演练评估与疫情处置复盘情况"
                    "定期修订（原则上每两年修订一次，必要时及时修订）。"
                )
            if slot_name == "实施与发布":
                return (
                    f"（AI校验修复）本预案自发布之日起实施。发生{dname}聚集性疫情时立即启动，"
                    "原相关预案同时废止或按最新要求执行。"
                )
            if slot_name == "附表与联系方式":
                return (
                    "（AI校验修复）联系人：校医院值班电话（待填），学生工作部电话（待填），"
                    "后勤保障处电话（待填）；属地疾控中心值班电话（待填），定点医院急救电话120。"
                )
            return ""

        for sec in sections or []:
            if not isinstance(sec, dict):
                continue
            title = self._strip_title_prefix(str(sec.get("title") or "")).strip()
            paras = sec.get("paragraphs")
            paras_list = paras if isinstance(paras, list) else []
            new_paras: List[str] = []
            for p in paras_list:
                s = str(p or "").strip()
                parsed = self._parse_slot_paragraph(s)
                if not parsed:
                    new_paras.append(s)
                    continue
                slot, value = parsed
                value_clean = self._strip_inline_outline_marks(value)
                value_clean = re.sub(
                    r"[（(]\s*智能体补充[^）)]*[）)]\s*", "", str(value_clean or "")
                ).strip()
                value_clean = re.sub(
                    r"[（(]\s*知识库未检索到[^）)]*[）)]\s*", "", value_clean
                ).strip()
                value_clean = re.sub(
                    r"^\s*[（(]\s*\d+\s*[）)]\s*", "", str(value_clean or "")
                ).strip()
                value_clean = re.sub(r"^\s*\d+[.、]\s*", "", value_clean).strip()
                value_clean = re.sub(
                    r"^\s*[一二三四五六七八九十]+[.、]\s*", "", value_clean
                ).strip()
                value_clean = re.sub(r"\s+", " ", value_clean).strip()
                value_clean = re.sub(r"[。．]{2,}", "。", value_clean).strip()
                must_terms, ban_terms = self._default_slot_terms(slot)
                has_kb_missing = "（知识库缺失）" in value_clean
                has_political = any(k in value_clean for k in political_terms)
                place_mismatch = False
                if is_school and slot in {
                    "值班与报告链路",
                    "联动机制",
                    "指挥体系与职责",
                }:
                    place_mismatch = any(k in value_clean for k in school_place_bad)
                if is_school and slot == "监测对象与指标":
                    has_school_terms = any(
                        k in value_clean
                        for k in [
                            "学生",
                            "师生",
                            "教职工",
                            "晨检",
                            "午检",
                            "因病缺勤",
                            "缺课",
                        ]
                    )
                    if not has_school_terms:
                        place_mismatch = True
                    if ("中小学校" in value_clean or "托幼" in value_clean) and (
                        "大学" not in value_clean
                        and (not loc_text or loc_text not in value_clean)
                    ):
                        place_mismatch = True
                    if (not has_school_terms) and any(
                        k in value_clean for k in ["污水", "排污口"]
                    ):
                        place_mismatch = True
                if is_school and slot == "报告流程与时限":
                    if any(k in value_clean for k in ["中学", "小学", "托幼"]) and (
                        "大学" not in value_clean
                    ):
                        place_mismatch = True
                    if (
                        "疾控" in value_clean
                        and not any(
                            k in value_clean
                            for k in ["学校", "校医院", "卫生所", "学生处"]
                        )
                        and (not loc_text or loc_text not in value_clean)
                    ):
                        place_mismatch = True
                if is_school and slot == "启动条件与分级":
                    if any(
                        k in value_clean
                        for k in [
                            "65岁",
                            "老年",
                            "基层医疗卫生机构",
                            "社区（村）",
                            "居（村）民委员会",
                            "不再判定密切接触者",
                            "不再划定高低风险区",
                        ]
                    ):
                        place_mismatch = True
                if is_school and slot == "处置流程":
                    if any(
                        k in value_clean
                        for k in [
                            "第一责任人",
                            "倡导公众",
                            "提高健康素养",
                            "全方位、多渠道开展",
                        ]
                    ):
                        place_mismatch = True
                if is_school and slot == "队伍与技术保障":
                    if any(
                        k in value_clean
                        for k in ["口岸", "入境", "基因组测序", "报送中国疾控中心"]
                    ):
                        place_mismatch = True
                    if any(
                        k in value_clean
                        for k in [
                            "中国疾病预防控制中心",
                            "省级疾病预防控制中心",
                            "县级以上疾病预防控制机构",
                        ]
                    ) and not any(
                        k in value_clean
                        for k in ["学校", "校医院", "卫生所", "后勤", "学生处"]
                    ):
                        place_mismatch = True
                if is_school and slot == "监督检查与记录":
                    if any(k in value_clean for k in ["卫生监督机构", "执法", "稽查"]):
                        place_mismatch = True
                if slot == "实施与发布":
                    if any(
                        k in value_clean
                        for k in [
                            "隔离措施",
                            "核酸",
                            "抗原",
                            "密切接触者",
                            "高低风险区",
                        ]
                    ):
                        place_mismatch = True
                if slot == "附表与联系方式":
                    if any(
                        k in value_clean
                        for k in ["教育部", "moe.edu.cn", "体育卫生与艺术教育司"]
                    ):
                        place_mismatch = True
                need = (
                    has_kb_missing
                    or (not self._slot_semantic_ok(slot, value_clean))
                    or _mentions_other_disease(value_clean)
                    or (
                        slot in {"目的与适用范围", "编制依据"}
                        and _missing_allowed_disease(value_clean)
                    )
                    or place_mismatch
                    or has_political
                    or (
                        is_school
                        and slot in {"指挥体系与职责", "联动机制", "值班与报告链路"}
                        and any(mt and mt in value_clean for mt in macro_terms)
                    )
                )
                if need:
                    judge = await self._llm_judge_slot_value(
                        section_title=title,
                        slot_name=slot,
                        current_value=value_clean,
                        disease_type=disease_type,
                        location=location,
                    )
                    hard_invalid = False
                    hard_reason = ""
                    if place_mismatch:
                        hard_invalid = True
                        hard_reason = f"place_mismatch_{slot}"
                    if (
                        not self._slot_semantic_ok(slot, value_clean)
                    ) and not hard_invalid:
                        hard_invalid = True
                        hard_reason = f"slot_semantic_fail_{slot}"
                    if (
                        (not hard_invalid)
                        and is_school
                        and slot in {"指挥体系与职责", "联动机制", "值班与报告链路"}
                        and any(mt and mt in value_clean for mt in macro_terms)
                    ):
                        hard_invalid = True
                        hard_reason = f"macro_off_topic_{slot}"
                    if slot == "工作原则" and dt_norm != "food_poisoning":
                        if any(
                            k in value_clean
                            for k in ["食品安全", "体育卫生与艺术教育司", "报送我部"]
                        ):
                            hard_invalid = True
                            hard_reason = "work_principles_off_topic_food_safety"
                    judge_ok = bool(judge.get("ok"))
                    if judge_ok and not hard_invalid:
                        if value_clean and value_clean != value:
                            new_paras.append(f"{slot}：{value_clean}")
                            repairs.append(
                                {
                                    "section": title,
                                    "slot": slot,
                                    "old": value,
                                    "new": value_clean,
                                    "reason": "format_normalize",
                                    "judge_reason": str(judge.get("reason") or ""),
                                }
                            )
                            continue
                        new_paras.append(s)
                        continue
                    if allow_terms and slot in {"目的与适用范围", "编制依据"}:
                        must_terms = list(must_terms or []) + list(allow_terms or [])
                    evidence = self._pick_evidence(
                        guidelines=guidelines,
                        keywords=[title, slot, disease_type, location],
                        must_terms=must_terms,
                        prefer_terms=[disease_type, location],
                        ban_terms=ban_terms,
                        top_n=3,
                        slot_name=slot,
                        location=location,
                    )
                    rw = await self._llm_rewrite_slot_value(
                        section_title=title,
                        slot_name=slot,
                        current_value=value_clean,
                        disease_type=disease_type,
                        location=location,
                        evidence=evidence,
                    )
                    revised = str(rw.get("revised") or "").strip()
                    if not revised:
                        revised = _fallback_slot(slot)
                    if revised:
                        revised = self._strip_inline_outline_marks(revised)
                        revised = re.sub(r"\s+", " ", revised).strip()
                        revised = re.sub(
                            r"(^|[。；;\n])\s*[（(]\s*\d+\s*[）)]\s*", r"\1", revised
                        )
                        revised = re.sub(
                            r"^[，,。；;:：、!?！？—-]+\s*", "", revised
                        ).strip()
                        new_paras.append(f"{slot}：{revised}")
                        repairs.append(
                            {
                                "section": title,
                                "slot": slot,
                                "old": value,
                                "judge_reason": str(judge.get("reason") or ""),
                                "new": revised,
                                "reason": hard_reason or str(rw.get("reason") or ""),
                            }
                        )
                        continue
                if value_clean and value_clean != value:
                    new_paras.append(f"{slot}：{value_clean}")
                    repairs.append(
                        {
                            "section": title,
                            "slot": slot,
                            "old": value,
                            "new": value_clean,
                            "reason": "format_normalize",
                        }
                    )
                    continue
                new_paras.append(s)
            sec2 = dict(sec)
            sec2["title"] = title
            sec2["paragraphs"] = new_paras
            out.append(sec2)
        return out, repairs

    @staticmethod
    def _clean_excerpt(excerpt: str) -> str:
        s = str(excerpt or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[（(]\s*来源\s*[:：][^）)]*[）)]", "", s)
        s = re.sub(r"[（(]\s*注\s*[:：][^）)]*[）)]", "", s)
        s = re.sub(r"^[，,。；;:：)\]】》】]+", "", s)
        s = re.sub(r"^[（(]\s*\d+\s*[）)]\s*", "", s)
        s = re.sub(r"^\d+[.、]\s*", "", s)
        s = re.sub(r"^[（(]?[一二三四五六七八九十]+[）)]\s*", "", s)
        s = re.sub(r"^\s*(第\s*)?\d+\s*(章|节|条)\s*", "", s)
        s = re.sub(r"^\s*(第\s*)?[一二三四五六七八九十]+\s*(章|节|条)\s*", "", s)
        s = re.sub(r"\b\d+(?:\.\d+)+\s*", "", s)
        heading_terms = [
            "总则",
            "总体要求",
            "总体原则",
            "指导思想",
            "指导原则",
            "工作原则",
            "目的与适用范围",
            "编制依据",
            "组织",
            "指挥",
            "职责",
            "监测",
            "预警",
            "报告",
            "应急",
            "响应",
            "后期处置",
            "保障措施",
            "附则",
        ]
        head_alt = "|".join([re.escape(t) for t in heading_terms if t])
        if head_alt:
            s = re.sub(rf"^\s*\d+\s*(?:{head_alt})\s*", "", s)
            s = re.sub(rf"^\s*(?:{head_alt})\s*", "", s)
            s = re.sub(rf"\b\d+\s*(?:{head_alt})\b", "", s)
        s = PlanValidationAgent._strip_inline_outline_marks(s)
        return s.strip()

    @classmethod
    def _excerpt_to_paragraph(
        cls,
        slot_name: str,
        g: Dict[str, Any],
        *,
        must_terms: Optional[List[str]] = None,
        prefer_terms: Optional[List[str]] = None,
        ban_terms: Optional[List[str]] = None,
    ) -> str:
        excerpt = cls._clean_excerpt(str(g.get("excerpt") or ""))
        if not excerpt:
            return ""
        must_terms = must_terms or []
        prefer_terms = prefer_terms or []
        ban_terms = ban_terms or []

        parts = re.split(r"[。；;]\s*", excerpt)
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            text = excerpt
        else:
            bad_prefix = (
                "的",
                "并",
                "或",
                "以及",
                "与",
                "，",
                ",",
                "（",
                "(",
                "）",
                ")",
            )
            scored: List[tuple[int, str]] = []
            for p in parts[:12]:
                if len(p) < 10:
                    continue
                if p.startswith(bad_prefix):
                    continue
                score = 0
                for t in must_terms:
                    t2 = str(t or "").strip()
                    if t2 and t2 in p:
                        score += 3
                for t in prefer_terms:
                    t2 = str(t or "").strip()
                    if t2 and t2 in p:
                        score += 1
                for t in ban_terms:
                    t2 = str(t or "").strip()
                    if t2 and t2 in p:
                        score -= 6
                scored.append((score, p))

            if scored and must_terms:
                scored.sort(key=lambda x: x[0], reverse=True)
                selected = [p for s, p in scored if s > 0][:2]
                if not selected:
                    selected = [p for _, p in scored[:2]]
                text = "。".join(selected).strip()
                if must_terms and not any(t and t in text for t in must_terms):
                    return ""
            else:
                start_idx = 0
                for i, p in enumerate(parts[:6]):
                    if len(p) < 12:
                        continue
                    if p.startswith(bad_prefix):
                        continue
                    start_idx = i
                    break
                text = "。".join(parts[start_idx : start_idx + 2]).strip()
        text = cls._strip_inline_outline_marks(text)
        text = re.sub(r"^[，,。；;:：、!?！？—-]+\s*", "", text).strip()
        if not text:
            return ""
        if not cls._slot_semantic_ok(slot_name, text):
            return ""
        if len(text) > 180:
            text = text[:180].rstrip()
        if text and text[-1] not in "。！？””）)":
            text += "。"
        return f"{slot_name}：{text}"

    @classmethod
    def _polish_sections(cls, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_title = set()
        for sec in sections or []:
            if not isinstance(sec, dict):
                continue
            title = cls._strip_title_prefix(str(sec.get("title") or "")).strip()
            if not title or title in seen_title:
                continue
            seen_title.add(title)
            paras = sec.get("paragraphs")
            paras_list = paras if isinstance(paras, list) else []
            clean_paras = []
            seen_p = set()
            for p in paras_list:
                s = str(p or "").strip()
                s = re.sub(r"\s+", " ", s)
                s = s.replace("：，", "：")
                s = s.replace("：,", "：")
                s = re.sub(r"：\s*[，,。；;:：、!?！？—-]+\s*", "：", s)
                s = re.sub(r"[（(]\s*来源\s*[:：][^）)]*[）)]", "", s).strip()
                s = re.sub(
                    r"\b\d+\s*(总则|总体要求|工作原则|编制依据)\b", "", s
                ).strip()
                s = cls._strip_inline_outline_marks(s)
                if not s:
                    continue
                if s in seen_p:
                    continue
                seen_p.add(s)
                clean_paras.append(s)
            out.append(
                {
                    "title": title,
                    "paragraphs": clean_paras,
                    "subsections": sec.get("subsections") or [],
                }
            )
        return out

    async def _generate_missing_paragraphs(
        self,
        section_title: str,
        slot_name: str,
        disease_type: str,
        location: str,
        evidence: List[Dict[str, Any]],
    ) -> str:
        prompt = (
            "你是疾控预案编制助手。请为指定章节的指定槽位生成一段可执行的预案内容。\n"
            "要求：\n"
            '- 输出严格 JSON（不要输出多余文本），格式：{"paragraph": "..."}\n'
            "- paragraph 必须为中文，且不少于 30 字\n"
            "- 必须包含：责任主体、触发条件或时限、流程要点、联动对象（至少 2 个要素）\n"
            "- 内容必须结合 disease_type 与 location 场景\n"
            "- 如果 evidence 非空，请优先沿用 evidence 的措辞与约束；如果 evidence 为空，允许合理生成但避免空泛\n"
        )
        ctx = {
            "section_title": section_title,
            "slot": slot_name,
            "disease_type": disease_type,
            "location": location,
            "evidence": [
                {
                    "source_file": e.get("source_file"),
                    "excerpt": e.get("excerpt"),
                }
                for e in (evidence or [])[:3]
                if isinstance(e, dict)
            ],
        }
        raw = await self.llm.ask(
            messages=[Message.user_message(json.dumps(ctx, ensure_ascii=False))],
            system_msgs=[Message.system_message(prompt)],
            stream=False,
            temperature=0.2,
        )
        obj = _extract_json(raw)
        p = str(obj.get("paragraph") or "").strip()
        if len(p) >= 20:
            return p
        return (
            f"建议：由{location}疫情防控工作专班牵头，围绕“{slot_name}”制定具体流程与责任分工，"
            f"并与属地疾控与定点医院建立联动机制，确保在规定时限内完成处置。"
        )

    async def _build_required_sections_from_guidelines(
        self,
        disease_type: str,
        location: str,
        guidelines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        required = [
            (
                "总则",
                [
                    (
                        "目的与适用范围",
                        ["目的", "适用范围", "预案", "应急", "防控", "处置"],
                    ),
                    (
                        "编制依据",
                        ["依据", "方案", "指南", "技术", "要求", "条例", "传染病"],
                    ),
                    (
                        "工作原则",
                        ["原则", "预防为主", "科学", "依法", "联防", "联控", "分级"],
                    ),
                ],
            ),
            (
                "应急组织指挥体系",
                [
                    (
                        "指挥体系与职责",
                        ["指挥", "领导", "工作组", "职责", "专班", "分工"],
                    ),
                    (
                        "联动机制",
                        ["联动", "疾控", "医院", "教育", "卫生", "信息共享", "协同"],
                    ),
                    (
                        "值班与报告链路",
                        ["值班", "报告", "通报", "联系人", "24小时", "机制"],
                    ),
                ],
            ),
            (
                "监测、预警与报告",
                [
                    (
                        "监测对象与指标",
                        ["监测", "症状", "发热", "腹泻", "呕吐", "缺勤", "登记"],
                    ),
                    ("预警阈值", ["预警", "聚集性", "2例", "3例", "阈值", "标准"]),
                    (
                        "报告流程与时限",
                        ["报告", "上报", "直报", "时限", "2小时", "24小时"],
                    ),
                ],
            ),
            (
                "应急响应",
                [
                    (
                        "启动条件与分级",
                        ["启动", "响应", "分级", "条件", "风险等级", "触发"],
                    ),
                    (
                        "处置流程",
                        ["隔离", "就医", "消毒", "流调", "采样", "密接", "健康监测"],
                    ),
                    (
                        "风险沟通与信息发布",
                        ["通报", "信息发布", "舆情", "沟通", "宣传", "引导"],
                    ),
                ],
            ),
            (
                "后期处置",
                [
                    (
                        "响应终止条件",
                        ["终止", "解除", "结束", "条件", "连续", "无新增"],
                    ),
                    ("评估与复盘", ["评估", "总结", "复盘", "报告", "整改"]),
                    ("恢复与关怀", ["恢复", "复课", "复工", "心理", "疏导", "关怀"]),
                ],
            ),
            (
                "保障措施",
                [
                    (
                        "物资与经费保障",
                        ["物资", "储备", "调拨", "经费", "保障", "供应"],
                    ),
                    (
                        "队伍与技术保障",
                        ["队伍", "培训", "演练", "技术", "检测", "消毒"],
                    ),
                    ("监督检查与记录", ["监督", "检查", "记录", "台账", "评估"]),
                ],
            ),
            (
                "附则",
                [
                    ("解释与修订", ["解释", "修订", "更新", "评估", "改版"]),
                    ("实施与发布", ["实施", "发布", "生效", "适用", "执行"]),
                    (
                        "附表与联系方式",
                        ["联系方式", "联系人", "附表", "清单", "流程图"],
                    ),
                ],
            ),
        ]

        section_objs: List[Dict[str, Any]] = []
        report: List[Dict[str, Any]] = []
        disease_terms = _disease_query_terms(str(disease_type))
        place_terms = _place_query_terms(str(location))
        used_keys_global: set = set()
        ban_terms = ["蚊媒", "登革", "疟疾", "寨卡", "乙脑", "黄热"]
        dt_norm = str(disease_type).strip().lower()
        if dt_norm == "influenza":
            ban_terms += [
                "新冠",
                "SARS-CoV-2",
                "COVID-19",
                "covid",
                "诺如",
                "norovirus",
                "百日咳",
                "麻疹",
                "风疹",
                "结核",
                "结核病",
                "TB",
            ]
        elif dt_norm == "covid19":
            ban_terms += [
                "流感",
                "甲流",
                "influenza",
                "诺如",
                "norovirus",
                "百日咳",
                "麻疹",
                "风疹",
                "结核",
                "结核病",
                "TB",
            ]
        elif dt_norm == "norovirus":
            ban_terms += [
                "流感",
                "甲流",
                "influenza",
                "SARS-CoV-2",
                "COVID-19",
                "covid",
                "新冠",
                "新型冠状病毒感染防控方案",
                "麻疹",
                "风疹",
                "百日咳",
                "结核",
                "结核病",
                "TB",
            ]

        slot_min_score = {
            "预警阈值": 4,
            "报告流程与时限": 4,
            "值班与报告链路": 4,
            "监测对象与指标": 4,
            "启动条件与分级": 4,
            "响应终止条件": 4,
            "物资与经费保障": 4,
        }
        slot_must_terms: Dict[str, List[str]] = {
            "目的与适用范围": ["目的", "适用", "范围", "预案"],
            "编制依据": ["依据", "方案", "指南", "预案", "条例", "规范"],
            "工作原则": ["原则", "依法", "科学", "联防", "联控", "分级"],
            "指挥体系与职责": ["指挥", "职责", "负责", "协调", "组织"],
            "联动机制": ["联动", "协同", "机制", "协作", "部门"],
            "值班与报告链路": ["值班", "报告", "上报", "信息", "流程", "时限"],
            "预警阈值": ["阈值", "标准", "聚集", "预警"],
            "报告流程与时限": ["报告", "上报", "直报", "时限", "小时"],
            "监测对象与指标": ["监测", "病例", "症状", "缺勤", "登记"],
            "启动条件与分级": ["启动", "响应", "分级", "条件", "风险"],
            "处置流程": ["处置", "报告", "隔离", "消毒", "健康监测"],
            "风险沟通与信息发布": ["通报", "发布", "沟通", "宣传", "舆情"],
            "响应终止条件": ["终止", "解除", "条件", "无新增"],
            "评估与复盘": ["评估", "复盘", "总结", "整改"],
            "恢复与关怀": ["恢复", "复课", "复工", "关怀", "心理"],
            "物资与经费保障": ["物资", "储备", "调拨", "经费", "保障"],
            "队伍与技术保障": ["队伍", "培训", "演练", "技术", "检测"],
            "监督检查与记录": ["监督", "检查", "记录", "台账"],
            "解释与修订": ["解释", "修订", "更新"],
            "实施与发布": ["实施", "发布", "生效", "执行"],
            "附表与联系方式": ["联系方式", "联系人", "附表", "清单"],
        }
        slot_ban_terms: Dict[str, List[str]] = {}
        prefer_terms_global = (
            self._split_terms(disease_terms)
            + self._split_terms(place_terms)
            + self._split_terms(str(location))
        )
        if dt_norm == "influenza":
            slot_must_terms["处置流程"] = [
                "处置",
                "流调",
                "密接",
                "隔离",
                "就医",
                "消毒",
                "通风",
                "聚集性",
            ]
            slot_must_terms["监测对象与指标"] = [
                "监测",
                "流感",
                "流感样",
                "发热",
                "缺勤",
                "登记",
            ]
            slot_must_terms["报告流程与时限"] = ["报告", "上报", "直报", "时限", "小时"]
            slot_must_terms["启动条件与分级"] = ["启动", "响应", "条件", "分级"]
            slot_ban_terms = {
                "处置流程": ["煎服", "加减", "芦根", "藿香", "佩兰", "中医", "方剂"],
                "启动条件与分级": ["处警", "公安", "社会安全", "武警", "人防"],
                "值班与报告链路": [
                    "口岸",
                    "入境",
                    "全基因组",
                    "测序",
                    "新冠",
                    "百日咳",
                    "结核",
                    "诺如",
                    "麻疹",
                    "风疹",
                ],
                "监测对象与指标": ["全员核酸", "不再开展全员核酸", "核酸筛查"],
                "工作原则": ["结核", "结核病", "TB"],
                "预警阈值": ["流行病学史", "临床表现", "抗原检测阳性", "诊断标准"],
                "指挥体系与职责": ["应急管理部", "中央", "国务院", "党中央"],
            }
        elif dt_norm == "norovirus":
            slot_must_terms["监测对象与指标"] = [
                "监测",
                "呕吐",
                "腹泻",
                "缺勤",
                "聚集性",
            ]
            slot_must_terms["处置流程"] = [
                "处置",
                "病例",
                "流调",
                "采样",
                "消毒",
                "呕吐物",
                "手卫生",
            ]
            slot_must_terms["报告流程与时限"] = ["报告", "上报", "时限", "小时", "疾控"]
            slot_must_terms["预警阈值"] = [
                "阈值",
                "聚集",
                "暴发",
                "病例",
                "呕吐",
                "腹泻",
            ]
            slot_must_terms["指挥体系与职责"] = [
                "指挥",
                "职责",
                "乡镇",
                "疾控",
                "卫生院",
            ]
            slot_must_terms["联动机制"] = [
                "联动",
                "协同",
                "学校",
                "养老",
                "市场监管",
                "教育",
            ]
            slot_must_terms["值班与报告链路"] = [
                "值班",
                "报告",
                "疾控",
                "卫生院",
                "电话",
            ]
            slot_ban_terms = {
                "指挥体系与职责": [
                    "pH",
                    "地下水",
                    "冷藏",
                    "冷冻",
                    "乙醇",
                    "75%",
                    "60℃",
                    "发病机制",
                ],
                "值班与报告链路": ["发病机制", "尚不明确"],
            }
        used_text_global: set = set()

        for title, slots in required:
            paras: List[str] = []
            used_sources: List[str] = []
            missing_slots: List[str] = []
            used_keys_section: set = set()
            used_text_section: set = set()
            for slot_name, kws in slots:
                keywords = (
                    list(kws)
                    + self._split_terms(place_terms)
                    + self._split_terms(str(location))
                    + self._split_terms(disease_terms)
                    + self._split_terms(str(disease_type))
                )
                candidates = self._pick_evidence(
                    guidelines,
                    keywords=keywords,
                    must_terms=slot_must_terms.get(slot_name) or [],
                    slot_name=slot_name,
                    location=str(location),
                    top_n=6,
                    exclude_keys=used_keys_global | used_keys_section,
                    ban_terms=ban_terms,
                    min_score=int(slot_min_score.get(slot_name, 2)),
                )
                if not candidates:
                    candidates = self._pick_evidence(
                        guidelines,
                        keywords=keywords,
                        must_terms=slot_must_terms.get(slot_name) or [],
                        slot_name=slot_name,
                        location=str(location),
                        top_n=6,
                        exclude_keys=used_keys_section,
                        ban_terms=ban_terms,
                        min_score=int(slot_min_score.get(slot_name, 2)),
                    )
                picked = None
                picked_para = ""
                for cand in candidates:
                    excerpt_clean = self._clean_excerpt(str(cand.get("excerpt") or ""))
                    sig = excerpt_clean[:80]
                    if not sig:
                        continue
                    if sig in used_text_section or sig in used_text_global:
                        continue
                    must = slot_must_terms.get(slot_name) or []
                    if must and not any(mt and (mt in excerpt_clean) for mt in must):
                        continue
                    para = self._excerpt_to_paragraph(
                        slot_name,
                        cand,
                        must_terms=slot_must_terms.get(slot_name) or [],
                        prefer_terms=prefer_terms_global,
                        ban_terms=slot_ban_terms.get(slot_name) or [],
                    )
                    if not para:
                        continue
                    picked = cand
                    picked_para = para
                    used_text_section.add(sig)
                    used_text_global.add(sig)
                    break
                if picked:
                    e0 = picked
                    src = str(e0.get("source_file") or "").strip()
                    key = (src, str(e0.get("chunk_id") or ""))
                    used_keys_global.add(key)
                    used_keys_section.add(key)
                    if src:
                        used_sources.append(src)
                    paras.append(picked_para)
                    continue

                missing_slots.append(slot_name)
                if self.kb_only:
                    paras.append(
                        f"{slot_name}：（知识库缺失）未检索到直接依据，请补充与“{title}-{slot_name}”相关的规范条款后再生成。"
                    )
                else:
                    p = await self._generate_missing_paragraphs(
                        section_title=title,
                        slot_name=slot_name,
                        disease_type=str(disease_type),
                        location=str(location),
                        evidence=[],
                    )
                    p2 = str(p or "").strip()
                    if p2 and p2[-1] in "：:":
                        p2 = p2[:-1].strip()
                    paras.append(
                        f"{slot_name}：（智能体补充，知识库未检索到直接依据）{p2}"
                    )

            while len(paras) < 3:
                if self.kb_only:
                    paras.append(
                        f"（知识库缺失）请补充与“{title}-补充要求”相关的规范条款后再生成。"
                    )
                else:
                    p = await self._generate_missing_paragraphs(
                        section_title=title,
                        slot_name="补充要求",
                        disease_type=str(disease_type),
                        location=str(location),
                        evidence=[],
                    )
                    p2 = str(p or "").strip()
                    if p2 and p2[-1] in "：:":
                        p2 = p2[:-1].strip()
                    paras.append(
                        f"补充要求：（智能体补充，知识库未检索到直接依据）{p2}"
                    )

            section_objs.append(
                {
                    "title": title,
                    "paragraphs": paras,
                    "subsections": [],
                }
            )
            report.append(
                {
                    "title": title,
                    "used_sources": sorted(set([s for s in used_sources if s])),
                    "missing_slots": missing_slots,
                }
            )

        used_all = []
        for r in report:
            if isinstance(r, dict):
                used_all.extend(r.get("used_sources") or [])
        used_all = sorted(
            set([s for s in used_all if isinstance(s, str) and s.strip()])
        )
        if used_all:
            dt_norm = (
                str(getattr(disease_type, "value", disease_type) or "").strip().lower()
            )
            allow_terms = []
            if dt_norm == "influenza":
                allow_terms = ["流感", "流行性感冒", "influenza", "flu"]
            elif dt_norm == "norovirus":
                allow_terms = ["诺如", "诺如病毒", "norovirus", "诺瓦克"]
            elif dt_norm == "covid19":
                allow_terms = [
                    "新冠",
                    "新型冠状病毒",
                    "新型冠状病毒感染",
                    "covid",
                    "covid-19",
                    "sars-cov-2",
                ]
            generic_allow = [
                "传染病防治法",
                "突发公共卫生事件应急条例",
                "国家突发事件总体应急预案",
                "国家突发公共卫生事件应急预案",
                "学校卫生工作条例",
            ]
            other_disease_terms = [
                "新冠",
                "冠状病毒",
                "covid",
                "covid-19",
                "sars-cov-2",
                "诺如",
                "norovirus",
                "百日咳",
                "结核",
                "麻疹",
                "风疹",
            ]
            src_lines = []
            for s in used_all:
                s2 = str(s).strip()
                s2 = re.sub(r"^\s*[-—–]+\s*", "", s2).strip()
                if not s2:
                    continue
                s2_low = s2.lower()
                is_generic = any(t in s2 for t in generic_allow)
                is_allow = any(
                    t and (t in s2 or t.lower() in s2_low) for t in allow_terms
                )
                is_other = any(
                    t and (t in s2 or t.lower() in s2_low) for t in other_disease_terms
                )
                if is_other and not is_allow:
                    continue
                if not (is_generic or is_allow):
                    continue
                src_lines.append(f"- {s2}")
            section_objs.append(
                {
                    "title": "规范依据（章节来源）",
                    "paragraphs": [
                        "本预案“总则/组织体系/监测预警/应急响应/后期处置/保障措施/附则”等章节内容主要依据以下规范文件整理：",
                        *src_lines,
                    ],
                    "subsections": [],
                }
            )
        return {"sections": self._polish_sections(section_objs), "report": report}

    async def _inject_sections_rag_first(
        self,
        improved_plan_obj: Dict[str, Any],
        disease_type: str,
        location: str,
        guidelines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        built = await self._build_required_sections_from_guidelines(
            disease_type=str(disease_type),
            location=str(location),
            guidelines=guidelines,
        )
        required_sections = (
            built.get("sections") if isinstance(built.get("sections"), list) else []
        )
        report = built.get("report") if isinstance(built.get("report"), list) else []

        existing = improved_plan_obj.get("sections")
        existing_list = existing if isinstance(existing, list) else []

        dt_norm = str(disease_type or "").strip().lower()
        section_ban_terms: List[str] = []
        if dt_norm == "influenza":
            section_ban_terms = [
                "煎服",
                "加减",
                "芦根",
                "藿香",
                "佩兰",
                "方剂",
                "处警",
                "公安",
                "社会安全",
                "武警",
                "人防",
                "结核",
                "结核病",
                "TB",
                "百日咳",
                "诺如",
                "麻疹",
                "风疹",
            ]

        note_sections = []
        others = []
        existing_by_title: Dict[str, Dict[str, Any]] = {}
        seen_titles = set()
        for s in existing_list:
            if not isinstance(s, dict):
                continue
            t = str(s.get("title") or "").strip()
            t2 = self._strip_title_prefix(t)
            if t2 and t2 not in existing_by_title:
                existing_by_title[t2] = s
            if t2 == "人工修改说明":
                note_sections.append(s)
                continue
            seen_titles.add(t2)
            others.append(s)

        def _has_kb_missing(sec: Dict[str, Any]) -> bool:
            paras = sec.get("paragraphs")
            if not isinstance(paras, list):
                return False
            for p in paras:
                if "（知识库缺失）" in str(p):
                    return True
            return False

        def _contains_ban_terms(sec: Dict[str, Any]) -> bool:
            if not section_ban_terms:
                return False
            paras = sec.get("paragraphs")
            text = " ".join([str(p) for p in paras]) if isinstance(paras, list) else ""
            return any(bt and bt in text for bt in section_ban_terms)

        merged_required: List[Dict[str, Any]] = []
        for sec in required_sections:
            if not isinstance(sec, dict):
                continue
            t2 = self._strip_title_prefix(str(sec.get("title") or ""))
            existing_sec = existing_by_title.get(t2)
            if (
                existing_sec
                and isinstance(existing_sec, dict)
                and (self._is_section_shallow(sec) or _has_kb_missing(sec))
                and (not self._is_section_shallow(existing_sec))
                and self._section_slots_semantic_ok(t2, existing_sec.get("paragraphs"))
                and (not _contains_ban_terms(existing_sec))
            ):
                merged_required.append(existing_sec)
            else:
                merged_required.append(sec)

        required_titles = [
            self._strip_title_prefix(s.get("title"))
            for s in merged_required
            if isinstance(s, dict)
        ]
        filtered_others = []
        for s in others:
            t2 = self._strip_title_prefix(str(s.get("title") or ""))
            if t2 in required_titles:
                continue
            filtered_others.append(s)

        hardened_others: List[Dict[str, Any]] = []
        keep_titles = {"资源调配与缺口"}
        for s in filtered_others:
            if not isinstance(s, dict):
                continue
            title = self._strip_title_prefix(str(s.get("title") or ""))
            if self.kb_only and title not in keep_titles:
                continue
            paras = s.get("paragraphs")
            if isinstance(paras, list) and paras:
                s = dict(s)
                if title in keep_titles:
                    s["paragraphs"] = [
                        f"（系统计算）{str(p).strip()}" for p in paras if str(p).strip()
                    ]
                else:
                    s["paragraphs"] = [
                        f"（智能体补充，知识库未检索到直接依据）{str(p).strip()}"
                        for p in paras
                        if str(p).strip()
                    ]
            hardened_others.append(s)

        repaired_required, semantic_repairs = await self._semantic_repair_sections(
            sections=merged_required,
            disease_type=str(disease_type),
            location=str(location),
            guidelines=guidelines,
        )
        improved_plan_obj["sections"] = self._polish_sections(
            note_sections + repaired_required + hardened_others
        )
        return {
            "improved_plan": improved_plan_obj,
            "sections_fill_report": report,
            "semantic_repairs": semantic_repairs,
        }

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
        else:
            required_detail_titles = {
                "总则",
                "应急组织指挥体系",
                "监测、预警与报告",
                "应急响应",
                "后期处置",
                "保障措施",
                "附则",
            }
            for i, sec in enumerate(plan.sections or []):
                title = PlanValidationAgent._strip_title_prefix(
                    getattr(sec, "title", "")
                )
                if title not in required_detail_titles:
                    continue
                paras = getattr(sec, "paragraphs", None) or []
                if not isinstance(paras, list):
                    paras = []
                if len(paras) < 3 or any(
                    not PlanValidationAgent._is_paragraph_detailed(p) for p in paras
                ):
                    issues.append(
                        {
                            "type": "shallow_section",
                            "field": f"sections[{i}]",
                            "message": f"章节“{title}”内容过于空泛，需要补充具体可执行内容（至少 3 段，且每段不少于 20 字）",
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
        disease_type = getattr(disease_type, "value", disease_type)

        search_tool = CDCGuidelineSearchTool()
        disease_terms = _disease_query_terms(disease_type)
        place_terms = _place_query_terms(location)
        query = (
            f"{disease_terms} {place_terms} 防控 方案 技术 指南 预案 规范 要求 "
            "总则 组织 指挥 监测 预警 报告 应急响应 后期处置 保障 附则 审批 签字"
        )
        search_res = await search_tool.execute(
            query=query, disease_type=str(disease_type), top_k=20, mode="auto"
        )
        retrieved_guidelines = []
        if search_res.output and isinstance(search_res.output, str):
            try:
                search_data = json.loads(search_res.output)
                retrieved_guidelines = search_data.get("results", [])
            except Exception:
                pass
        if str(disease_type).strip().lower() == "influenza":
            extra_queries = [
                f"{disease_terms} 信息报告 直报 时限 2小时 24小时 传染病 信息报告管理 规范",
                f"{disease_terms} 学校 缺勤 聚集性 预警 阈值 停课 报告",
            ]
            for q in extra_queries:
                extra_res = await search_tool.execute(
                    query=q, disease_type=str(disease_type), top_k=20, mode="auto"
                )
                if extra_res.output and isinstance(extra_res.output, str):
                    try:
                        extra_data = json.loads(extra_res.output)
                        extra_list = extra_data.get("results", [])
                        if isinstance(extra_list, list) and extra_list:
                            retrieved_guidelines.extend(extra_list)
                    except Exception:
                        pass
            dedup = {}
            for g in retrieved_guidelines or []:
                if not isinstance(g, dict):
                    continue
                src = str(g.get("source_file") or "").strip()
                cid = g.get("chunk_id")
                key = (src, str(cid))
                if not src or cid is None:
                    continue
                if key not in dedup:
                    dedup[key] = g
            retrieved_guidelines = list(dedup.values())[:40]

        improve_prompt = (
            "你是疾控预案校验智能体。根据结构化校验错误(validation_errors)、硬性规则问题(rule_issues)，以及检索到的疾控规范(guidelines)，对预案进行深度业务校验。\n"
            "输出严格 JSON（不要输出多余文本）。\n"
            "字段：valid(布尔), summary(中文一段话), issues(数组), suggestions(数组), thinking_summary(中文要点，列 3-6 条), improved_plan(对象，可选)。\n"
            "要求：\n"
            "1. issues 需结合输入中的 errors/issues，以及是否符合检索到的规范要求来归纳。\n"
            "2. suggestions 给出可执行的修改要点，若存在规范偏离需重点指出。\n"
            "3. 请输出 improved_plan（作为优化后的草稿预案）。若原预案已较完整，也请在 improved_plan 中给出优化后的版本（可在原基础上小幅完善）。improved_plan 必须符合 CDCPlanDocument 结构，不要额外字段。\n"
            "4. improved_plan.measures[].citations 必须是对象数组，每个对象包含 source_file, chunk_id, score, excerpt；并且必须从给定 guidelines 中选择（不要编造）。\n"
            "5. thinking_summary 必须说明如何结合规范知识库发现了问题并给出了建议。\n"
            "6. improved_plan.sections 必须包含并补充具体可执行内容（不要只写一句话）：总则、应急组织指挥体系、监测、预警与报告、应急响应、后期处置、保障措施、附则。\n"
            "   - 标题请使用不带序号的中文标题（例如“总则”而不是“一、总则”），以便前端/导出端统一重编号。\n"
            "   - 每个章节至少 3 段 paragraphs，每段不少于 20 字，必须包含“责任主体/触发条件/流程要点/时限要求/联动对象”等具体信息。\n"
            "   - 内容必须结合 disease_type 与 location 场景（如大学流感：呼吸道传播、人群聚集、校医院/教务/后勤联动；诺如：食堂饮水、呕吐物处置、肠道隔离等）。\n"
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
                original_measures = (
                    plan_obj.get("measures")
                    if isinstance(plan_obj, dict)
                    and isinstance(plan_obj.get("measures"), list)
                    else None
                )
                original_risk = (
                    plan_obj.get("risk")
                    if isinstance(plan_obj, dict)
                    and isinstance(plan_obj.get("risk"), dict)
                    else None
                )
                original_input = (
                    plan_obj.get("input")
                    if isinstance(plan_obj, dict)
                    and isinstance(plan_obj.get("input"), dict)
                    else None
                )
                repair_prompt = (
                    "你是疾控预案校验智能体。请修复 improved_plan，使其满足结构校验与规则校验。\n"
                    '只输出严格 JSON（不要输出多余文本），格式：{"improved_plan": {...}}。\n'
                    "2. improved_plan.measures[].level 只能为 core 或 supplementary。\n"
                    "3. 所有 core 措施必须至少包含 1 条 citations。\n"
                    "4. citations 必须为对象数组，每个对象包含 source_file, chunk_id, score, excerpt。\n"
                    "5. citations 必须从给定 guidelines 中选择（source_file+chunk_id 必须匹配），禁止编造。\n"
                    "6. improved_plan.sections 不得为空；并补充“总则/应急组织指挥体系/监测、预警与报告/应急响应/后期处置/保障措施/附则”等章节的具体内容（每章至少 3 段）。\n"
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
                    if original_measures is not None:
                        candidate["measures"] = original_measures
                    if original_risk is not None:
                        candidate["risk"] = original_risk
                    if isinstance(original_input, dict):
                        cand_input = candidate.get("input")
                        if isinstance(cand_input, dict):
                            rp = original_input.get("region_profile")
                            if rp is not None and str(rp).strip():
                                cand_input["region_profile"] = str(rp).strip()
                                candidate["input"] = cand_input
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

        sections_fill_report: List[Dict[str, Any]] = []
        if not isinstance(improved_plan_obj, dict):
            improved_plan_obj = plan_obj if isinstance(plan_obj, dict) else {}
        if isinstance(improved_plan_obj, dict):
            meta = improved_plan_obj.get("meta")
            if isinstance(meta, dict) and meta.get("title"):
                t0 = str(meta.get("title") or "")
                t1 = self._sanitize_plan_title(t0)
                if t1:
                    meta["title"] = t1
                    improved_plan_obj["meta"] = meta
            injected = await self._inject_sections_rag_first(
                improved_plan_obj=improved_plan_obj,
                disease_type=str(disease_type),
                location=str(location),
                guidelines=retrieved_guidelines,
            )
            improved_plan_obj = (
                injected.get("improved_plan")
                if isinstance(injected.get("improved_plan"), dict)
                else improved_plan_obj
            )
            sections_fill_report = (
                injected.get("sections_fill_report")
                if isinstance(injected.get("sections_fill_report"), list)
                else []
            )
            try:
                CDCPlanDocument(**improved_plan_obj)
                improved_plan_validation_errors = []
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
                "sections_fill_report": sections_fill_report,
            },
        }
        self.memory.add_message(
            Message.assistant_message(json.dumps(result, ensure_ascii=False, indent=2))
        )
        self.state = AgentState.FINISHED
        return json.dumps(result, ensure_ascii=False, indent=2)
