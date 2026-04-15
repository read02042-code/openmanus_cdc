import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkRecord:
    chunk_id: int
    source_file: str
    start: int
    end: int
    text: str


@dataclass
class SearchResult:
    score: float
    source_file: str
    chunk_id: int
    text: str


class GuidelineVectorStore:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Disable logging and progress bars during standard runs
        import logging
        import warnings

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="sentence_transformers"
        )
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.records: List[ChunkRecord] = []

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join(line.rstrip() for line in text.split("\n")).strip()

    def _chunk_text(self, text: str) -> List[tuple[int, int, str]]:
        normalized = self._normalize_text(text)
        chunks: List[tuple[int, int, str]] = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(normalized), step):
            end = min(start + self.chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append((start, end, chunk))
            if end >= len(normalized):
                break
        return chunks

    def _collect_chunks(self, raw_dir: Path) -> List[ChunkRecord]:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        txt_files = sorted(raw_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt guideline files found in: {raw_dir}")

        records: List[ChunkRecord] = []
        chunk_id = 0
        for txt_file in txt_files:
            content = txt_file.read_text(encoding="utf-8")
            for start, end, chunk in self._chunk_text(content):
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source_file=txt_file.name,
                        start=start,
                        end=end,
                        text=chunk,
                    )
                )
                chunk_id += 1
        return records

    def build(self, raw_dir: Path) -> None:
        self.records = self._collect_chunks(raw_dir)
        texts = [record.text for record in self.records]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save(self, index_dir: Path) -> None:
        if self.index is None or not self.records:
            raise ValueError("Index has not been built yet. Call build() first.")

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "guideline.faiss"))
        (index_dir / "meta.json").write_text(
            json.dumps(
                [asdict(record) for record in self.records],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (index_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "metric": "cosine (IndexFlatIP + normalized embeddings)",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "GuidelineVectorStore":
        config_path = index_dir / "config.json"
        meta_path = index_dir / "meta.json"
        faiss_path = index_dir / "guideline.faiss"
        if (
            not config_path.exists()
            or not meta_path.exists()
            or not faiss_path.exists()
        ):
            raise FileNotFoundError(
                f"Index files not complete under: {index_dir}. "
                "Expected guideline.faiss, meta.json and config.json."
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))
        store = cls(
            model_name=config["model_name"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        store.records = [
            ChunkRecord(**r) for r in json.loads(meta_path.read_text(encoding="utf-8"))
        ]
        store.index = faiss.read_index(str(faiss_path))
        return store

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.index is None:
            raise ValueError("Index is not loaded. Build or load index first.")
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        top_k = min(top_k, len(self.records))
        scores, ids = self.index.search(query_vec, top_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            record = self.records[idx]
            results.append(
                SearchResult(
                    score=float(score),
                    source_file=record.source_file,
                    chunk_id=record.chunk_id,
                    text=record.text,
                )
            )
        return results


def _build_cmd(args: argparse.Namespace) -> None:
    store = GuidelineVectorStore(
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    store.build(Path(args.raw_dir))
    store.save(Path(args.index_dir))
    print(f"Index built successfully under: {args.index_dir}")


def _search_cmd(args: argparse.Namespace) -> None:
    store = GuidelineVectorStore.load(Path(args.index_dir))
    results = store.search(args.query, top_k=args.top_k)
    print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CDC guideline FAISS vector store")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="Build FAISS index from raw guidelines")
    build_parser.add_argument("--raw-dir", default="knowledage/raw")
    build_parser.add_argument("--index-dir", default="knowledage/faiss_index")
    build_parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    build_parser.add_argument("--chunk-size", type=int, default=500)
    build_parser.add_argument("--chunk-overlap", type=int, default=100)
    build_parser.set_defaults(func=_build_cmd)

    search_parser = sub.add_parser("search", help="Search guideline chunks by query")
    search_parser.add_argument("--index-dir", default="knowledage/faiss_index")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.set_defaults(func=_search_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
