"""Hybrid index: BM25 (lexical) + LSA dense vectors (semantic-ish).

Why LSA instead of a downloaded neural encoder: the whole repo must run
offline on the bundled Python, and a 256-dim LSA over character n-grams keeps
exact financial terms and numbers while still giving a dense vector for fuzzy
matches. The dense side is deliberately *not* called an LLM embedding.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401,E402

import joblib  # noqa: E402
import jieba  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from src.config import INDEX, META, PROCESSED  # noqa: E402

jieba.setLogLevel(20)
STOPWORDS = set(
    "的 了 和 与 及 或 是 在 为 对 中 上 下 年 月 日 公司 本 报告 期 内 元 万元 亿元 一 二 三 四 五 六 七 八 九 十".split()
)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = []
    for w in jieba.lcut_for_search(text):
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if len(w) == 1 and not w.isdigit() and not re.match(r"[a-z]", w):
            continue
        toks.append(w)
    toks.extend(NUM_RE.findall(text))
    return toks


@dataclass
class Hit:
    idx: int
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float = 0.0
    bm25_score: float = 0.0


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        self.doc_tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for doc in docs:
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.doc_tf.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.doc_tf), dtype=np.float32)
        q = set(query_tokens)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-6))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


class HybridIndex:
    def __init__(self, chunks, vectorizer, svd, embeddings, bm25, meta):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.svd = svd
        self.embeddings = embeddings
        self.bm25 = bm25
        self.meta = meta

    @classmethod
    def build(cls, chunks_path: Path = PROCESSED / "chunks.jsonl") -> "HybridIndex":
        chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not chunks:
            raise SystemExit(f"no chunks at {chunks_path}; run src/extract.py first")
        print(f"building index for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=2,
            max_features=180_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X = vectorizer.fit_transform(texts)
        n_comp = min(256, X.shape[1] - 1, max(32, len(chunks) - 1))
        print(f"tf-idf {X.shape}, svd -> {n_comp} dims")
        svd = TruncatedSVD(n_components=n_comp, random_state=42, algorithm="randomized", n_iter=7)
        emb = svd.fit_transform(X).astype(np.float32)
        emb = normalize(emb).astype(np.float32)

        print("tokenizing for BM25...")
        docs = [tokenize(t) for t in texts]
        bm25 = BM25(docs)

        meta = {
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "chunks": len(chunks),
            "companies": len({c["code"] for c in chunks}),
            "dense_dim": int(emb.shape[1]),
            "tfidf_features": int(X.shape[1]),
            "avg_tokens": round(float(bm25.avgdl), 2),
            "tokenizer": "jieba.lcut_for_search + numeric spans",
            "dense": "char_wb 2-4gram TF-IDF -> TruncatedSVD(256) -> L2",
        }
        idx = cls(chunks, vectorizer, svd, emb, bm25, meta)
        idx.save()
        return idx

    def save(self, out: Path = INDEX) -> None:
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out / "vectorizer.joblib")
        joblib.dump(self.svd, out / "svd.joblib")
        np.save(out / "embeddings.npy", self.embeddings)
        with open(out / "bm25.pkl", "wb") as f:
            pickle.dump(
                {
                    "k1": self.bm25.k1,
                    "b": self.bm25.b,
                    "doc_len": self.bm25.doc_len,
                    "avgdl": self.bm25.avgdl,
                    "doc_tf": self.bm25.doc_tf,
                    "df": self.bm25.df,
                    "idf": self.bm25.idf,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        (out / "index_meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved index -> {out}")

    @classmethod
    def load(cls, path: Path = INDEX) -> "HybridIndex":
        chunks = [json.loads(line) for line in (PROCESSED / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        vectorizer = joblib.load(path / "vectorizer.joblib")
        svd = joblib.load(path / "svd.joblib")
        emb = np.load(path / "embeddings.npy")
        with open(path / "bm25.pkl", "rb") as f:
            state = pickle.load(f)
        if isinstance(state, dict):
            bm25 = BM25.__new__(BM25)
            bm25.k1 = state["k1"]
            bm25.b = state["b"]
            bm25.doc_len = state["doc_len"]
            bm25.avgdl = state["avgdl"]
            bm25.doc_tf = state["doc_tf"]
            bm25.df = state["df"]
            bm25.idf = state["idf"]
        else:
            bm25 = state
        meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
        return cls(chunks, vectorizer, svd, emb, bm25, meta)

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        qv = self.svd.transform(q).astype(np.float32)
        qv = normalize(qv)
        return (self.embeddings @ qv.T).ravel()

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(tokenize(query))

    def search(
        self,
        query: str,
        top_k: int = 12,
        company_codes: list[str] | None = None,
        rrf_k: int = 60,
        dense_weight: float = 0.55,
        bm25_weight: float = 0.45,
        dense_pool: int = 120,
        bm25_pool: int = 120,
    ) -> list[Hit]:
        n = len(self.chunks)
        allowed = None
        if company_codes:
            allowed = {i for i, c in enumerate(self.chunks) if c["code"] in set(company_codes)}
            if not allowed:
                return []
        dense = self.dense_scores(query)
        bm = self.bm25_scores(query)
        if allowed is not None:
            mask = np.zeros(n, dtype=bool)
            mask[list(allowed)] = True
            dense = np.where(mask, dense, -np.inf)
            bm = np.where(mask, bm, -np.inf)

        d_top = np.argsort(-dense)[: min(dense_pool, n)]
        b_top = np.argsort(-bm)[: min(bm25_pool, n)]
        d_rank = {int(i): r + 1 for r, i in enumerate(d_top) if np.isfinite(dense[i])}
        b_rank = {int(i): r + 1 for r, i in enumerate(b_top) if np.isfinite(bm[i])}

        hits = []
        for i in set(d_rank) | set(b_rank):
            if allowed is not None and i not in allowed:
                continue
            s = 0.0
            if i in d_rank:
                s += dense_weight / (rrf_k + d_rank[i])
            if i in b_rank:
                s += bm25_weight / (rrf_k + b_rank[i])
            hits.append(
                Hit(
                    idx=i,
                    score=s,
                    dense_rank=d_rank.get(i),
                    bm25_rank=b_rank.get(i),
                    dense_score=float(dense[i]) if np.isfinite(dense[i]) else 0.0,
                    bm25_score=float(bm[i]) if np.isfinite(bm[i]) else 0.0,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return self._dedupe(hits, top_k)

    @staticmethod
    def _dedupe(hits: list[Hit], top_k: int) -> list[Hit]:
        """Drop near-duplicate chunks from the same page (table + prose overlap)."""
        out: list[Hit] = []
        seen_pages: dict[tuple[str, int, str], int] = {}
        for h in hits:
            c = None
            # caller-facing dedupe below needs chunks; handled in Retriever
            out.append(h)
            if len(out) >= top_k * 3:
                break
        return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default=str(PROCESSED / "chunks.jsonl"))
    args = parser.parse_args()
    idx = HybridIndex.build(Path(args.chunks))
    print(json.dumps(idx.meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
