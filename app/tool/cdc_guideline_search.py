import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from app.config import config
from app.tool.base import BaseTool, ToolResult


class CDCGuidelineSearchTool(BaseTool):
    name: str = "cdc_guideline_search"
    description: str = (
        "Search CDC guideline knowledge base and return top matched excerpts with source and score."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "disease_type": {
                "type": "string",
                "description": "Optional disease/event type hint (e.g., influenza, norovirus, covid19)",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
            "mode": {
                "type": "string",
                "description": "auto: try vector search then fallback; faiss: vector only; keyword: keyword only",
                "enum": ["auto", "faiss", "keyword"],
                "default": "auto",
            },
            "index_dir": {
                "type": "string",
                "description": "Index directory, relative to project root by default",
                "default": "knowledage/faiss_index",
            },
            "raw_dir": {
                "type": "string",
                "description": "Raw guideline directory, relative to project root by default",
                "default": "knowledage/raw",
            },
        },
        "required": ["query"],
    }

    _store: Optional[Any] = PrivateAttr(default=None)
    _loaded_from: Optional[str] = PrivateAttr(default=None)

    @staticmethod
    def _normalize_disease_type(value: str) -> str:
        s = str(value or "").strip().lower()
        s = re.sub(r"[\s_]+", "", s)
        if not s:
            return ""
        if s in {"flu", "influenza", "liugan", "liuxingxingmaogan"}:
            return "influenza"
        if s in {"norovirus", "nuorubingdu", "nuoru"}:
            return "norovirus"
        if s in {"covid", "covid19", "covid-19", "sarscov2", "sars-cov-2", "xg"}:
            return "covid19"
        if s in {"measles", "rubella", "mazhen", "fengzhen", "mazhenfengzhen"}:
            return "measles_rubella"
        if s in {"pertussis", "baerike"}:
            return "pertussis"
        if s in {"tuberculosis", "tb", "jiehe", "feijiehe"}:
            return "tuberculosis"
        if s in {"dengue", "dengre"}:
            return "dengue"
        if s in {"hfmd", "handfootmouth", "shouzukou", "shouzukoubing"}:
            return "hand_foot_mouth"
        if s in {"varicella", "shuadou"}:
            return "varicella"
        if s in {"mumps", "saixianyan"}:
            return "mumps"
        if s in {"hepatitisa", "jiahe", "jiaxingganyan"}:
            return "hepatitis_a"
        if s in {"foodpoisoning", "shiwu"}:
            return "food_poisoning"
        if "influenza" in s or "flu" in s or "流感" in s:
            return "influenza"
        if "norovirus" in s or "诺如" in s:
            return "norovirus"
        if "covid" in s or "新冠" in s or "冠状病毒" in s or "sarscov2" in s:
            return "covid19"
        if "麻疹" in s or "风疹" in s:
            return "measles_rubella"
        if "百日咳" in s:
            return "pertussis"
        if "结核" in s:
            return "tuberculosis"
        if "登革热" in s:
            return "dengue"
        if "手足口" in s:
            return "hand_foot_mouth"
        if "水痘" in s:
            return "varicella"
        if "腮腺炎" in s:
            return "mumps"
        if "甲肝" in s or "甲型肝炎" in s:
            return "hepatitis_a"
        if "食物中毒" in s:
            return "food_poisoning"
        return s

    @classmethod
    def _infer_disease_type_from_query(cls, query: str) -> str:
        q = str(query or "")
        if not q:
            return ""
        if re.search(r"\b(influenza|flu)\b", q, flags=re.I) or ("流感" in q):
            return "influenza"
        if re.search(r"\b(norovirus)\b", q, flags=re.I) or ("诺如" in q):
            return "norovirus"
        if re.search(r"\b(covid|covid-?19|sars-?cov-?2)\b", q, flags=re.I) or (
            "新冠" in q or "冠状病毒" in q
        ):
            return "covid19"
        if re.search(r"\b(measles|rubella)\b", q, flags=re.I) or ("麻疹" in q) or ("风疹" in q):
            return "measles_rubella"
        if re.search(r"\b(pertussis)\b", q, flags=re.I) or ("百日咳" in q):
            return "pertussis"
        if re.search(r"\b(tuberculosis|tb)\b", q, flags=re.I) or ("结核" in q):
            return "tuberculosis"
        if re.search(r"\b(dengue)\b", q, flags=re.I) or ("登革热" in q):
            return "dengue"
        if re.search(r"\b(hfmd|hand[-_ ]?foot[-_ ]?mouth)\b", q, flags=re.I) or (
            "手足口" in q
        ):
            return "hand_foot_mouth"
        if re.search(r"\b(varicella)\b", q, flags=re.I) or ("水痘" in q):
            return "varicella"
        if re.search(r"\b(mumps)\b", q, flags=re.I) or ("腮腺炎" in q):
            return "mumps"
        if re.search(r"\b(hepatitis\\s*a|hepatitis_a)\b", q, flags=re.I) or ("甲肝" in q):
            return "hepatitis_a"
        if ("食物中毒" in q) or re.search(r"\b(food[-_ ]?poisoning)\b", q, flags=re.I):
            return "food_poisoning"
        return ""

    @staticmethod
    def _disease_keywords(disease_type: str) -> Dict[str, List[str]]:
        dt = str(disease_type or "").strip().lower()
        return {
            "influenza": [
                "influenza",
                "flu",
                "流感",
                "流行性感冒",
                "甲型流感",
                "乙型流感",
                "H1N1",
                "H3N2",
                "流感样",
                "ILI",
            ],
            "norovirus": [
                "norovirus",
                "诺如",
                "诺如病毒",
                "诺瓦克",
                "NV",
            ],
            "covid19": [
                "covid",
                "covid-19",
                "sars-cov-2",
                "新冠",
                "新型冠状病毒",
                "新型冠状病毒感染",
                "新冠肺炎",
            ],
            "measles_rubella": ["麻疹", "风疹", "麻疹风疹", "measles", "rubella"],
            "pertussis": ["百日咳", "pertussis"],
            "tuberculosis": ["结核", "结核病", "肺结核", "tuberculosis", "tb"],
            "dengue": ["登革热", "dengue"],
            "hand_foot_mouth": ["手足口", "手足口病", "HFMD", "hand foot mouth"],
            "varicella": ["水痘", "varicella"],
            "mumps": ["腮腺炎", "流行性腮腺炎", "mumps"],
            "hepatitis_a": ["甲肝", "甲型肝炎", "hepatitis a", "hepatitis_a"],
            "food_poisoning": ["食物中毒", "食源性", "food poisoning", "food_poisoning"],
        }.get(dt, [])

    @classmethod
    def _other_disease_terms(cls, disease_type: str) -> List[str]:
        dt = str(disease_type or "").strip().lower()
        core = {
            "influenza": cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19")
            + ["百日咳", "结核", "麻疹", "风疹"],
            "norovirus": cls._disease_keywords("influenza")
            + cls._disease_keywords("covid19")
            + ["百日咳", "结核", "麻疹", "风疹"],
            "covid19": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + ["百日咳", "结核", "麻疹", "风疹"],
            "measles_rubella": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "pertussis": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "tuberculosis": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "dengue": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "hand_foot_mouth": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "varicella": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "mumps": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "hepatitis_a": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
            "food_poisoning": cls._disease_keywords("influenza")
            + cls._disease_keywords("norovirus")
            + cls._disease_keywords("covid19"),
        }.get(dt)
        return core or []

    @classmethod
    def _filter_and_rerank_by_disease(
        cls, results: List[Dict[str, Any]], disease_type: str, top_k: int
    ) -> List[Dict[str, Any]]:
        dt = cls._normalize_disease_type(disease_type)
        if not dt:
            return results[:top_k]
        allow = [t for t in cls._disease_keywords(dt) if t]
        ban = [t for t in cls._other_disease_terms(dt) if t]
        if not allow and not ban:
            return results[:top_k]

        rescored: List[Dict[str, Any]] = []
        strong_filtered: List[Dict[str, Any]] = []
        for r in results or []:
            if not isinstance(r, dict):
                continue
            src = str(r.get("source_file") or "")
            excerpt = str(r.get("excerpt") or "")
            hay = (src + "\n" + excerpt).lower()
            raw_score = r.get("score")
            try:
                base_score = float(raw_score) if raw_score is not None else 0.0
            except Exception:
                base_score = 0.0

            has_allow = any(t.lower() in hay for t in allow)
            has_ban = any(t.lower() in hay for t in ban)
            hard_mismatch = (not has_allow) and any(
                t.lower() in src.lower() for t in ban
            )
            if hard_mismatch:
                continue
            score = base_score
            if has_allow and has_ban:
                score = score * 0.8
            elif has_ban and not has_allow:
                score = score * 0.15
            elif has_allow:
                score = score * 1.05
            r2 = dict(r)
            r2["score"] = float(score)
            rescored.append(r2)
            if not (has_ban and not has_allow):
                strong_filtered.append(r2)

        if not rescored:
            return results[:top_k]
        min_keep = max(3, int(top_k // 2))
        pool = strong_filtered if len(strong_filtered) >= min_keep else rescored
        pool.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return pool[:top_k]

    @staticmethod
    def _as_abs_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return config.root_path / p

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
        if not normalized:
            return []
        step = max(1, chunk_size - chunk_overlap)
        chunks = []
        for start in range(0, len(normalized), step):
            end = min(start + chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append((start, end, chunk))
            if end >= len(normalized):
                break
        return chunks

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        q = str(query or "").strip()
        if not q:
            return []
        q = re.sub(r"[\[\]{}(),，。；;：:\n\r\t]+", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            return []
        toks: List[str] = []
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]{1,}|[\u4e00-\u9fff]{2,}", q):
            t = m.group(0).strip()
            if not t:
                continue
            toks.append(t)
        stop = {
            "方案",
            "指南",
            "规范",
            "要求",
            "预案",
            "工作",
            "处置",
            "应急",
            "管理",
            "措施",
            "技术",
            "防控",
        }
        out = []
        seen = set()
        for t in toks:
            if t in stop:
                continue
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            out.append(t)
        return out[:12]

    @staticmethod
    def _keyword_search(raw_dir: Path, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not raw_dir.exists():
            raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
        files = sorted(raw_dir.glob("*.txt"))
        if not files:
            return []

        q = query.strip()
        if not q:
            return []

        results: List[Dict[str, Any]] = []
        tokens = CDCGuidelineSearchTool._tokenize_query(q)
        q_lower = q.lower()
        for f in files:
            content = f.read_text(encoding="utf-8")
            if not content:
                continue
            for start, end, chunk in CDCGuidelineSearchTool._chunk_text(content):
                cl = chunk.lower()
                score = 0.0
                pos = cl.find(q_lower) if q_lower else -1
                if pos >= 0:
                    score += 2.0 / (1.0 + pos)
                for t in tokens:
                    if not t:
                        continue
                    tl = t.lower()
                    if tl in cl:
                        score += 1.0
                if score <= 0.0:
                    continue
                excerpt = chunk
                results.append(
                    {
                        "score": float(score),
                        "source_file": f.name,
                        "chunk_id": int(start),
                        "excerpt": excerpt,
                    }
                )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_store(self, index_dir: Path) -> Any:
        from app.rag import GuidelineVectorStore

        key = str(index_dir.resolve())
        if self._store is not None and self._loaded_from == key:
            return self._store
        store = GuidelineVectorStore.load(index_dir)
        self._store = store
        self._loaded_from = key
        return store

    @staticmethod
    def _to_tool_results(results: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                out.append(r)
                continue
            source_file = getattr(r, "source_file", "")
            chunk_id = getattr(r, "chunk_id", -1)
            score = getattr(r, "score", 0.0)
            text = getattr(r, "text", "")
            out.append(
                {
                    "score": float(score),
                    "source_file": str(source_file),
                    "chunk_id": int(chunk_id),
                    "excerpt": str(text),
                }
            )
        return out

    @staticmethod
    def _sanitize_query(query: str) -> str:
        q = (query or "").strip()
        q = re.sub(r"\s+", " ", q)
        return q

    async def execute(self, **kwargs) -> ToolResult:
        query = self._sanitize_query(kwargs.get("query", ""))
        disease_type = self._normalize_disease_type(kwargs.get("disease_type", ""))
        top_k = int(kwargs.get("top_k", 5) or 5)
        mode = str(kwargs.get("mode", "auto") or "auto").lower()
        index_dir = self._as_abs_path(kwargs.get("index_dir", "knowledage/faiss_index"))
        raw_dir = self._as_abs_path(kwargs.get("raw_dir", "knowledage/raw"))

        if not query:
            return ToolResult(error="query is required")
        if top_k < 1:
            top_k = 1
        if top_k > 20:
            top_k = 20

        if mode not in {"auto", "faiss", "keyword"}:
            return ToolResult(error="mode must be one of: auto, faiss, keyword")

        if mode == "keyword":
            try:
                results = self._keyword_search(raw_dir, query, top_k)
                dt2 = disease_type or self._infer_disease_type_from_query(query)
                results = self._filter_and_rerank_by_disease(results, dt2, top_k)
                return self.success_response({"query": query, "results": results})
            except Exception as e:
                return ToolResult(error=str(e))

        try:
            store = self._load_store(index_dir)
            results = store.search(query, top_k=top_k)
            tool_results = self._to_tool_results(results)
            dt2 = disease_type or self._infer_disease_type_from_query(query)
            tool_results = self._filter_and_rerank_by_disease(tool_results, dt2, top_k)
            return self.success_response({"query": query, "results": tool_results})
        except Exception as e:
            if mode == "faiss":
                return ToolResult(error=str(e))
            try:
                results = self._keyword_search(raw_dir, query, top_k)
                dt2 = disease_type or self._infer_disease_type_from_query(query)
                results = self._filter_and_rerank_by_disease(results, dt2, top_k)
                return self.success_response(
                    {
                        "query": query,
                        "results": results,
                        "fallback": "keyword",
                        "faiss_error": str(e),
                    }
                )
            except Exception as e2:
                return ToolResult(error=f"{str(e)}; fallback failed: {str(e2)}")
