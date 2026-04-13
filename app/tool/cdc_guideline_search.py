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
        q_lower = q.lower()
        for f in files:
            content = f.read_text(encoding="utf-8")
            if not content:
                continue
            for start, end, chunk in CDCGuidelineSearchTool._chunk_text(content):
                pos = chunk.lower().find(q_lower)
                if pos < 0:
                    continue
                score = 1.0 / (1.0 + pos)
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
                return self.success_response({"query": query, "results": results})
            except Exception as e:
                return ToolResult(error=str(e))

        try:
            store = self._load_store(index_dir)
            results = store.search(query, top_k=top_k)
            return self.success_response(
                {"query": query, "results": self._to_tool_results(results)}
            )
        except Exception as e:
            if mode == "faiss":
                return ToolResult(error=str(e))
            try:
                results = self._keyword_search(raw_dir, query, top_k)
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
