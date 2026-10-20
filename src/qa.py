"""Retrieval + extractive QA over the 白酒 annual-report corpus.

No LLM is called. Answers are built from retrieved rows/sentences, and every
claim carries a page-level citation. This makes the failure modes inspectable:
if retrieval misses the table, the answer says so instead of hallucinating.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401,E402

from src.config import BY_CODE, COMPANIES, METRIC_SYNONYMS  # noqa: E402
from src.index import HybridIndex  # noqa: E402

COMPANY_ALIASES = []
for c in COMPANIES:
    for a in c.text_aliases:
        COMPANY_ALIASES.append((a, c))
COMPANY_ALIASES.sort(key=lambda x: len(x[0]), reverse=True)

PANORAMA_WORDS = ("各家", "各公司", "12家", "十二家", "全部", "排名", "前三", "对比", "谁", "哪些公司", "行业", "全景", "最高", "最低", "第一")
QUAL_WORDS = ("为什么", "原因", "驱动", "战略", "风险", "措施", "如何", "怎么看", "解释", "主要系", "因为")
TREND_WORDS = ("同比", "增长", "下降", "变化", "增速", "增幅", "降幅")

METRIC_ANCHORS = {
    "营业总收入": "合并利润表 营业总收入 2024 年度",
    "营业收入": "主要会计数据 营业收入 2024 年度",
    "归母净利润": "主要会计数据 归属于上市公司股东的净利润 2024 年度",
    "扣非净利润": "主要会计数据 扣除非经常性损益的净利润 2024 年度",
    "经营活动现金流": "主要会计数据 经营活动产生的现金流量净额 2024 年度",
    "合同负债": "合并资产负债表 合同负债 期末余额 2024 年度",
    "研发投入": "研发投入合计 研发投入金额 2024 年度",
    "毛利率": "分行业 酒类 毛利率 2024 年度",
    "销售费用": "合并利润表 销售费用 2024 年度",
}


@dataclass
class QuestionSpec:
    original: str
    companies: list[str]
    metrics: list[str]
    years: list[int]
    intent: str
    panorama: bool


@dataclass
class Fact:
    company: str
    code: str
    metric: str
    value: float
    unit: str
    prior: float | None
    yoy: float | None
    year: int
    chunk_id: str
    section: str
    page: int
    snippet: str
    confidence: float
    value_yi: float | None = None

    def money_text(self) -> str:
        if self.unit == "%":
            return f"{self.value:.2f}%"
        v = self.value_yi if self.value_yi is not None else self.value / 1e8
        return f"{v:,.2f}亿元"


def detect_companies(query: str) -> list[str]:
    found: list[str] = []
    taken: list[tuple[int, int]] = []
    for alias, company in COMPANY_ALIASES:
        if company.code in found:
            continue
        start = query.find(alias)
        if start < 0:
            continue
        end = start + len(alias)
        if any(not (end <= a or start >= b) for a, b in taken):
            continue
        taken.append((start, end))
        found.append(company.code)
    return found


def detect_metrics(query: str) -> list[str]:
    hits: list[tuple[int, int, str]] = []
    for canonical, syns in METRIC_SYNONYMS.items():
        for syn in sorted(syns, key=len, reverse=True):
            start = query.find(syn)
            if start < 0:
                continue
            end = start + len(syn)
            if any(not (end <= a or start >= b) for a, b in [(x[0], x[1]) for x in hits]):
                continue
            hits.append((start, end, canonical))
            break
    return list(dict.fromkeys(h[2] for h in sorted(hits)))


def analyze(query: str) -> QuestionSpec:
    companies = detect_companies(query)
    metrics = detect_metrics(query)
    years = [int(y) for y in re.findall(r"(20\d{2})", query)]
    panorama = any(w in query for w in PANORAMA_WORDS) or len(companies) >= 3
    if any(w in query for w in QUAL_WORDS) and not any(w in query for w in ("排名", "前三", "最高", "最低")):
        intent = "qualitative"
    elif panorama and any(w in query for w in ("增速", "增长最快", "增幅", "降幅", "下滑", "跌幅")):
        intent = "growth"
    elif panorama and any(w in query for w in ("排名", "最高", "最低", "前三", "谁", "第一")):
        intent = "ranking"
    elif panorama or len(companies) >= 2:
        intent = "comparison"
    elif any(w in query for w in TREND_WORDS):
        intent = "trend"
    else:
        intent = "fact"
    return QuestionSpec(query, companies, metrics, years or [2024], intent, panorama)


# ---------------------------------------------------------------------------
# table / prose fact extraction
# ---------------------------------------------------------------------------


def _num(s: str) -> float | None:
    s = s.replace(",", "").replace("，", "").replace("%", "").strip()
    if not s or s in ("-", "—", "不适用", "/"):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _val_cells(cells: list[str]) -> list[tuple[int, float, bool]]:
    out = []
    for i, cell in enumerate(cells):
        v = _num(cell)
        if v is None:
            continue
        note = ("," not in cell and "." not in cell and abs(v) < 1000)
        out.append((i, v, note))
    return out


def parse_pipe(text: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return [], []
    raw = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        # PDF cells sometimes split a number: "3,978,131,52 8.88"
        cells = [re.sub(r"(?<=\d)\s+(?=[\d,])", "", c) for c in cells]
        raw.append(cells)
    # CNINFO grids often contain many empty spacer columns. Compact each row
    # independently; logical column order is preserved.
    rows = [[c for c in r if c] for r in raw]
    if not rows:
        return [], []
    header = rows[0]
    body = [r for r in rows[2:] if any(r)]

    # merge label rows split by a page/line break
    merged: list[list[str]] = []
    cont_re = re.compile(r"^(的|之|利润|净利润|净额|量|金额|（|\(|）|\))")
    for row in body:
        label = row[0].strip() if row else ""
        nonempty = [c for c in row if c.strip()]
        if merged and label and len(nonempty) == 1 and len(label) <= 24 and cont_re.match(label):
            prev = merged[-1]
            prev_nonempty = [c for c in prev if c.strip()]
            has_number = any(_num(c) is not None for c in prev_nonempty[1:])
            if has_number and len(prev_nonempty) >= 2:
                prev[0] = (prev[0] + label).strip()
                continue
        merged.append(row)
    return header, merged


def _norm_label(s: str) -> str:
    s = re.sub(r"^[一二三四五六七八九十]+、", "", s)
    s = re.sub(r"^其中[:：]", "", s)
    return s.replace(" ", "").replace("\u3000", "")


def _metric_syns(metric: str) -> list[str]:
    return sorted(METRIC_SYNONYMS.get(metric, [metric]), key=len, reverse=True)


def _find_row(body: list[list[str]], metric: str) -> tuple[int, int] | None:
    best = None
    for i, row in enumerate(body):
        if not row:
            continue
        label = _norm_label(row[0])
        if not label:
            continue
        score = 0
        for syn in _metric_syns(metric):
            s = _norm_label(syn)
            if label == s:
                score = max(score, 100)
            elif label.startswith(s) or label.endswith(s):
                score = max(score, 80)
            elif s in label:
                score = max(score, 60)
            elif len(label) >= 8 and s.startswith(label):
                # split label: "归属于上市公司股东的" + "净利润（元）"
                score = max(score, 72)
        if metric == "归母净利润":
            if "归属于上市公司股东" in label:
                score += 50
            elif label == "净利润":
                score -= 35
        if metric == "研发投入":
            if "研发投入合计" in label:
                score += 45
            elif "研发投入" in label:
                score += 20
            elif "研发费用" in label:
                score -= 15
        if score and (best is None or score > best[1]):
            best = (i, score)
    return best


def _header_mapping(header: list[str]) -> tuple[int | None, int | None, int | None, int | None]:
    cur = prior = yoy = margin = None
    for i, h in enumerate(header):
        h = h.replace(" ", "")
        if cur is None and ("2024" in h or "本期" in h or "期末" in h) and "上年" not in h and "上期" not in h:
            cur = i
        if prior is None and ("2023" in h or "上期" in h or "上年" in h or "期初" in h):
            prior = i
        if yoy is None and ("增减" in h or "同比" in h or "变动" in h):
            yoy = i
        if margin is None and "毛利率" in h:
            margin = i
    return cur, prior, yoy, margin


def _unit_from(text: str) -> str:
    m = re.search(r"单位[:：]\s*(亿元|万元|元)", text)
    if m:
        return m.group(1)
    return "元"


def _align_offset(
    header: list[str], row: list[str], cur: int | None, prior: int | None, yoy: int | None
) -> int:
    """Find the offset that maps header year columns onto this row's values.

    CNINFO tables often drop the "附注" cell or the leading label cell, so the
    header and body can be shifted by one. We score each shift by how many
    year columns land on plausible numbers.
    """
    if cur is None:
        return 0
    best_o, best_score = 0, -1e9
    for o in range(-3, 4):
        ci, pi, yi = cur + o, (prior + o if prior is not None else None), (yoy + o if yoy is not None else None)
        if ci < 0 or ci >= len(row):
            continue
        cv = _num(row[ci])
        if cv is None:
            continue
        cell = row[ci]
        score = 0.0
        if "," in cell or "." in cell:
            score += 2
        if abs(cv) >= 1000:
            score += 2
        if 1900 <= abs(cv) <= 2100 and "," not in cell and "." not in cell:
            score -= 8
        if pi is not None and 0 <= pi < len(row) and _num(row[pi]) is not None:
            score += 4
        if yi is not None and 0 <= yi < len(row) and _num(row[yi]) is not None:
            score += 1
        score -= abs(o) * 0.15
        if score > best_score:
            best_o, best_score = o, score
    return best_o


def extract_table_fact(chunk: dict, metric: str) -> Fact | None:
    header, body = parse_pipe(chunk.get("raw_text", ""))
    if not body:
        return None
    if "第一季度" in chunk.get("raw_text", "")[:1200] or "分季度" in (chunk.get("page_hint") or ""):
        return None
    if metric == "毛利率":
        raw = chunk.get("raw_text", "")
        if "毛利率" not in raw:
            return None
        if not any("毛利率" in h for h in header) and not ("营业成本" in raw and "营业收入" in raw):
            return None
    found = _find_row(body, metric)
    cur_col, prior_col, yoy_col, margin_col = _header_mapping(header)
    if not found and metric == "毛利率":
        for i, row in enumerate(body):
            label = _norm_label(row[0]) if row else ""
            if margin_col is not None and margin_col < len(row) and ("酒类" in label or "小计" in label or "合计" in label):
                found = (i, 70)
                break
    if not found:
        return None
    ri, row_score = found
    row = body[ri]
    nums = _val_cells(row)
    significant = [(i, v) for i, v, note in nums if not note]
    if not significant:
        return None
    money_count = sum(1 for i, v, note in nums if "%" not in row[i] and not note)
    if metric != "毛利率" and cur_col is None and money_count >= 4:
        # four quarters packed into one row, with no year header
        return None
    unit = _unit_from(chunk["text"])
    value = prior = yoy = None

    if metric == "毛利率":
        unit = "%"
        # A real gross-margin column is labelled "毛利率", not
        # "毛利率比上年增减"; try the labels nearest the header positions.
        exact_cols = [i for i, h in enumerate(header) if "毛利率" in h and "比上年" not in h and "增减" not in h]
        for col in exact_cols or ([margin_col] if margin_col is not None else []):
            for off in (0, -1, 1, -2, 2):
                idx = col + off
                if 0 <= idx < len(row) and ("%" in row[idx] or col in exact_cols):
                    v = _num(row[idx])
                    if v is not None and 0 <= v <= 100:
                        value = v
                        break
            if value is not None:
                break
        if value is None:
            # header rows were split by the PDF: pick the largest percentage,
            # which is the margin rather than a YoY growth rate.
            pcts = [_num(c) for c in row[2:] if "%" in c]
            pcts = [p for p in pcts if p is not None and 0 <= p <= 100]
            value = max(pcts) if pcts else None
    else:
        off = _align_offset(header, row, cur_col, prior_col, yoy_col)
        value_idx = None
        if cur_col is not None and 0 <= cur_col + off < len(row):
            value = _num(row[cur_col + off])
            if value is not None:
                value_idx = cur_col + off
        if value is None and significant:
            value = significant[0][1]
            value_idx = significant[0][0]

        # prior: prefer the mapped year cell, but never accept a ratio cell
        if prior_col is not None and 0 <= prior_col + off < len(row) and "%" not in row[prior_col + off]:
            prior = _num(row[prior_col + off])
        if prior is None:
            for i, v, note in nums:
                if i == value_idx or "%" in row[i] or note:
                    continue
                # skip small integers (note references, share counts, etc.)
                if abs(v) < 1000 and "," not in row[i]:
                    continue
                prior = v
                break
        yoy = None
        if prior not in (None, 0):
            yoy = (value - prior) / abs(prior) * 100
        elif yoy_col is not None and 0 <= yoy_col + off < len(row):
            yoy = _num(row[yoy_col + off])

    if value is None:
        return None

    value_yi = value / 1e4 if unit == "万元" else (value / 1e8 if unit == "元" else value)
    snippet = " | ".join(c for c in row if c)[:260]
    caption = chunk.get("caption") or ""
    confidence = float(row_score)
    if "主要会计" in caption:
        confidence += 12
    if "合并利润表" in caption:
        confidence += 8
    if "母公司" in caption:
        confidence -= 20
    if chunk["page_start"] <= 10 and "公司简介" in chunk["section"]:
        confidence += 10
    if metric == "营业收入" and chunk["page_start"] <= 10 and "公司简介" in chunk["section"]:
        confidence += 25
    if metric == "营业收入" and chunk["page_start"] <= 30 and "管理层讨论" in chunk["section"]:
        confidence += 20
    if metric == "营业收入" and "母公司" in caption:
        confidence -= 15
    if chunk["page_start"] <= 30 and "管理层讨论" in chunk["section"]:
        confidence += 6
    if "研发" in caption:
        confidence += 10
    if metric == "毛利率" and any("毛利率" in h for h in header):
        confidence += 12
    return Fact(
        company=chunk["company"],
        code=chunk["code"],
        metric=metric,
        value=value,
        unit=unit,
        prior=prior,
        yoy=yoy,
        year=chunk.get("year", 2024),
        chunk_id=chunk["chunk_id"],
        section=chunk["section"],
        page=chunk["page_start"],
        snippet=snippet,
        confidence=confidence,
        value_yi=value_yi if unit != "%" else None,
    )


def extract_prose_fact(chunk: dict, metric: str) -> Fact | None:
    text = re.sub(r"\s+", " ", chunk.get("raw_text", ""))
    for syn in _metric_syns(metric):
        for m in re.finditer(re.escape(syn), text):
            window = text[m.start(): m.start() + 130]
            mm = re.search(r"([\d,]+(?:\.\d+)?)\s*(亿元|万元|元|%)", window)
            if not mm:
                continue
            if metric == "毛利率" and mm.group(2) != "%":
                continue
            value = _num(mm.group(1))
            if value is None:
                continue
            unit = mm.group(2)
            yoy = None
            tail = text[m.start(): m.start() + 220]
            ym = re.search(r"同比(?:增长|下降|变动|增加|减少)?\s*([\-]?\d+(?:\.\d+)?)\s*%", tail)
            if ym:
                yoy = float(ym.group(1))
                if "下降" in ym.group(0) or "减少" in ym.group(0):
                    yoy = -abs(yoy)
            value_yi = value / 1e4 if unit == "万元" else (value / 1e8 if unit == "元" else value)
            return Fact(
                company=chunk["company"],
                code=chunk["code"],
                metric=metric,
                value=value,
                unit=unit,
                prior=None,
                yoy=yoy,
                year=chunk.get("year", 2024),
                chunk_id=chunk["chunk_id"],
                section=chunk["section"],
                page=chunk["page_start"],
                snippet=window[:240],
                confidence=50.0,
                value_yi=value_yi if unit != "%" else None,
            )
    return None


def fact_from_chunk(chunk: dict, metric: str) -> Fact | None:
    if chunk.get("content_type") == "table":
        f = extract_table_fact(chunk, metric)
        if f:
            return f
    return extract_prose_fact(chunk, metric)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


@dataclass
class Retrieved:
    chunk: dict
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float
    bm25_score: float
    company_rank: int | None = None

    def to_dict(self) -> dict:
        c = self.chunk
        raw = c["raw_text"]
        if c.get("content_type") == "table":
            rows = []
            for ln in raw.splitlines():
                if not ln.strip() or re.fullmatch(r"\|[\s\-|]+\|?", ln.strip()):
                    continue
                cells = [p.strip() for p in ln.strip("|").split("|") if p.strip()]
                if cells:
                    rows.append(" · ".join(cells))
                if len(rows) >= 3:
                    break
            snippet = " ／ ".join(rows)
        else:
            snippet = raw
        return {
            "chunk_id": c["chunk_id"],
            "company": c["company"],
            "code": c["code"],
            "section": c["section"],
            "page": c["page_start"],
            "type": c["content_type"],
            "caption": c.get("caption", ""),
            "score": round(self.score, 6),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_score": round(self.dense_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "company_rank": self.company_rank,
            "snippet": re.sub(r"\s+", " ", snippet)[:260],
        }


class Retriever:
    def __init__(self, index: HybridIndex):
        self.index = index

    @staticmethod
    def _table_boost(items: list[Retrieved], query: str) -> list[Retrieved]:
        """Numeric questions need the table row, not just the surrounding prose."""
        metrics = detect_metrics(query)
        if not metrics:
            return items
        syns = []
        for m in metrics:
            syns.extend(METRIC_SYNONYMS.get(m, [m]))
        for it in items:
            c = it.chunk
            if c.get("content_type") != "table":
                continue
            text = c.get("text", "")
            bonus = 0.0
            if any(s in text for s in syns):
                bonus += 0.0025
            if "2024" in text or "本期" in text or "期末" in text:
                bonus += 0.0012
            caption = c.get("caption") or ""
            if any(w in caption for w in ("主要会计", "利润表", "资产负债表", "研发投入")):
                bonus += 0.0015
            if c["page_start"] <= 10 and "公司简介" in c["section"]:
                bonus += 0.002
            elif c["page_start"] <= 30 and "管理层讨论" in c["section"]:
                bonus += 0.0015
            it.score += bonus
        items.sort(key=lambda x: x.score, reverse=True)
        return items

    def _enrich(self, hits) -> list[Retrieved]:
        return [
            Retrieved(
                chunk=self.index.chunks[h.idx],
                score=h.score,
                dense_rank=h.dense_rank,
                bm25_rank=h.bm25_rank,
                dense_score=h.dense_score,
                bm25_score=h.bm25_score,
            )
            for h in hits
        ]

    @staticmethod
    def _dedupe(items: list[Retrieved], limit: int, per_company: int | None = None) -> list[Retrieved]:
        out: list[Retrieved] = []
        seen_keys = set()
        per: dict[str, int] = {}
        seen_texts: list[str] = []
        for it in items:
            c = it.chunk
            key = (c["code"], c["page_start"], c["content_type"])
            if key in seen_keys:
                continue
            if per_company is not None and per.get(c["code"], 0) >= per_company:
                continue
            norm = re.sub(r"\s+", "", c.get("raw_text", ""))
            if norm and any(norm[:120] == t[:120] for t in seen_texts):
                continue
            seen_keys.add(key)
            seen_texts.append(norm)
            per[c["code"]] = per.get(c["code"], 0) + 1
            out.append(it)
            if len(out) >= limit:
                break
        return out

    def search(
        self,
        query: str,
        top_k: int = 12,
        company_codes: list[str] | None = None,
        panorama: bool = False,
    ) -> tuple[list[Retrieved], str]:
        if panorama:
            codes = company_codes or [c.code for c in COMPANIES]
            merged: list[Retrieved] = []
            for code in codes:
                hits = self.index.search(query, top_k=8, company_codes=[code], dense_pool=80, bm25_pool=80)
                enriched = self._table_boost(self._enrich(hits), query)
                for rank, it in enumerate(enriched, 1):
                    it.company_rank = rank
                    merged.append(it)
            merged.sort(key=lambda x: (x.company_rank or 99, -x.score))
            picked = self._dedupe(merged, limit=max(top_k, len(codes) * 3), per_company=3)
            return picked[:top_k], "panorama"
        hits = self.index.search(query, top_k=top_k * 4, company_codes=company_codes)
        items = self._table_boost(self._enrich(hits), query)
        items = self._dedupe(items, limit=top_k)
        return items, "single" if company_codes else "open"


# ---------------------------------------------------------------------------
# answer synthesis
# ---------------------------------------------------------------------------


def _fmt_yoy(y: float | None) -> str:
    if y is None:
        return "未披露/未能抽取"
    return f"{y:+.2f}%"


def best_fact(retriever: Retriever, company_code: str, metric: str) -> tuple[Fact | None, list[Retrieved]]:
    company = BY_CODE[company_code]
    q = f"{company.name} {METRIC_ANCHORS.get(metric, metric + ' 2024 主要会计数据 合并报表')}"
    hits, _ = retriever.search(q, top_k=24, company_codes=[company_code])
    facts = []
    for it in hits:
        f = fact_from_chunk(it.chunk, metric)
        if f:
            facts.append(f)
    if not facts:
        return None, hits
    facts.sort(key=lambda f: (-f.confidence, f.page))
    return facts[0], hits


def _citation_from_fact(f: Fact, n: int) -> dict:
    return {
        "n": n,
        "company": f.company,
        "code": f.code,
        "section": f.section,
        "page": f.page,
        "chunk_id": f.chunk_id,
        "snippet": f.snippet,
        "pdf_url": f"/pdf/{f.code}#page={f.page}",
    }


def _evidence_fallback(hits: list[Retrieved], limit: int = 3) -> list[dict]:
    return [
        {
            "text": re.sub(r"\s+", " ", h.chunk["raw_text"])[:300],
            "chunk_id": h.chunk["chunk_id"],
            "company": h.chunk["company"],
            "code": h.chunk["code"],
            "section": h.chunk["section"],
            "page": h.chunk["page_start"],
        }
        for h in hits[:limit]
    ]


def answer_question(question: str, retriever: Retriever, top_k: int = 12) -> dict:
    spec = analyze(question)
    codes = spec.companies
    panorama = spec.panorama
    search_query = question
    if spec.metrics and spec.intent != "qualitative":
        search_query += " " + " ".join(METRIC_ANCHORS.get(m, m) for m in spec.metrics)
    hits, mode = retriever.search(search_query, top_k=top_k, company_codes=codes or None, panorama=panorama)

    result = {
        "question": question,
        "spec": {
            "companies": [BY_CODE[c].name for c in spec.companies],
            "metrics": spec.metrics,
            "years": spec.years,
            "intent": spec.intent,
            "panorama": spec.panorama,
        },
        "answer_status": "ok",
        "headline": "",
        "bullets": [],
        "table": None,
        "citations": [],
        "retrieval": {
            "mode": mode,
            "query": search_query,
            "companies": codes or ([c.code for c in COMPANIES] if panorama else []),
            "chunks": [h.to_dict() for h in hits],
        },
        "notes": [],
    }

    if not spec.metrics or spec.intent == "qualitative":
        quotes = []
        for h in hits:
            text = re.sub(r"\s+", " ", h.chunk["raw_text"])
            sents = re.split(r"(?<=[。！？；])", text)
            keep = [s for s in sents if any(w in s for w in ("原因", "主要系", "由于", "风险", "措施", "战略", "同比"))]
            if not keep:
                keep = sents[:2]
            for s in keep[:2]:
                if len(s.strip()) >= 20:
                    quotes.append((s.strip()[:220], h))
            if len(quotes) >= 3:
                break
        if not quotes:
            quotes = [(re.sub(r"\s+", " ", h.chunk["raw_text"])[:220], h) for h in hits[:2]]
        result["headline"] = "未识别到单一财务指标，下面是与问题最相关的原文证据（抽取式，未改写）。"
        result["bullets"] = [f"{q} [{i + 1}]" for i, (q, _) in enumerate(quotes)]
        result["citations"] = [
            {
                "n": i + 1,
                "company": h.chunk["company"],
                "code": h.chunk["code"],
                "section": h.chunk["section"],
                "page": h.chunk["page_start"],
                "chunk_id": h.chunk["chunk_id"],
                "snippet": q,
                "pdf_url": f"/pdf/{h.chunk['code']}#page={h.chunk['page_start']}",
            }
            for i, (q, h) in enumerate(quotes)
        ]
        return result

    target_codes = codes or ([c.code for c in COMPANIES] if panorama else [])
    if not target_codes:
        target_codes = list(dict.fromkeys(h.chunk["code"] for h in hits[:3]))

    facts: list[Fact] = []
    fallback_notes = []
    lookup_map: dict[str, list[Retrieved]] = {}
    for code in target_codes:
        for metric in spec.metrics:
            f, fh = best_fact(retriever, code, metric)
            if not f and metric == "营业总收入":
                f, fh = best_fact(retriever, code, "营业收入")
                if f:
                    f.metric = metric
                    fallback_notes.append(BY_CODE[code].name)
            if f:
                facts.append(f)
                lookup_map[f.chunk_id] = fh
    if fallback_notes:
        result["notes"].append("以下公司未抽取到营业总收入，改用营业收入口径：" + "、".join(sorted(set(fallback_notes))))

    uniq: dict[tuple[str, str], Fact] = {}
    for f in facts:
        key = (f.code, f.metric)
        if key not in uniq or f.confidence > uniq[key].confidence:
            uniq[key] = f
    facts = list(uniq.values())
    fact_chunks = []
    seen_fact = set()
    for f in facts:
        if f.chunk_id in seen_fact:
            continue
        seen_fact.add(f.chunk_id)
        fh = lookup_map.get(f.chunk_id, [])
        hit = next((h for h in fh if h.chunk["chunk_id"] == f.chunk_id), None)
        if hit:
            d = hit.to_dict()
        else:
            d = {
                "chunk_id": f.chunk_id,
                "company": f.company,
                "code": f.code,
                "section": f.section,
                "page": f.page,
                "type": "table",
                "snippet": f.snippet,
            }
        d["fact_metric"] = f.metric
        d["fact_value"] = f.money_text()
        fact_chunks.append(d)
    result["retrieval"]["fact_chunks"] = fact_chunks

    if not facts:
        result["answer_status"] = "failed"
        result["headline"] = "没有检索到可核验的数值。下面是召回的相关段落，答案应以原文为准。"
        fb = _evidence_fallback(hits)
        result["bullets"] = [f"{e['text']} " for e in fb]
        result["citations"] = [
            {
                "n": i + 1,
                "company": e["company"],
                "code": e["code"],
                "section": e["section"],
                "page": e["page"],
                "chunk_id": e["chunk_id"],
                "snippet": e["text"],
                "pdf_url": f"/pdf/{e['code']}#page={e['page']}",
            }
            for i, e in enumerate(fb)
        ]
        return result

    if spec.intent == "growth":
        primary = spec.metrics[0]
        rows = [f for f in facts if f.metric == primary and f.yoy is not None]
        want_low = any(w in question for w in ("最低", "最慢", "降幅最大", "下滑最多"))
        want_high = any(w in question for w in ("最高", "最快", "增长最快", "增幅最大"))
        rows.sort(key=lambda f: f.yoy, reverse=not (want_low and not want_high))
        if rows:
            hi = max(rows, key=lambda f: f.yoy)
            lo = min(rows, key=lambda f: f.yoy)
            if want_low and want_high:
                result["headline"] = (
                    f"{primary}同比增速最高的是{hi.company}（{_fmt_yoy(hi.yoy)}），"
                    f"最低的是{lo.company}（{_fmt_yoy(lo.yoy)}）；共抽取到 {len(rows)}/{len(target_codes)} 家。"
                )
            elif want_low:
                result["headline"] = f"{primary}同比增速最低的是{lo.company}（{_fmt_yoy(lo.yoy)}）。"
            else:
                result["headline"] = f"{primary}同比增速最高的是{hi.company}（{_fmt_yoy(hi.yoy)}）。"
        table_rows, citations = [], []
        for i, f in enumerate(rows, 1):
            table_rows.append([str(i), f.company, f.money_text(), _fmt_yoy(f.yoy), str(f.page)])
            citations.append(_citation_from_fact(f, i))
        result["table"] = {"columns": ["排名", "公司", primary, "同比", "页码"], "rows": table_rows}
        bullets = []
        for i, f in enumerate(rows[:3], 1):
            bullets.append(f"第{i}名 {f.company}：{f.money_text()}，同比{_fmt_yoy(f.yoy)} [{i}]")
        if len(rows) > 3:
            i = len(rows)
            f = rows[-1]
            bullets.append(f"末位 {f.company}：{f.money_text()}，同比{_fmt_yoy(f.yoy)} [{i}]")
        result["bullets"] = bullets
        result["citations"] = citations
        if len(rows) < len(target_codes):
            result["answer_status"] = "partial"
            missing = [BY_CODE[c].name for c in target_codes if c not in {f.code for f in rows}]
            result["notes"].append(f"未抽取到同比：{'、'.join(missing)}")
        return result

    if panorama or len(target_codes) >= 2:
        primary = spec.metrics[0]
        rows = [f for f in facts if f.metric == primary]
        rows.sort(key=lambda f: f.value_yi if f.value_yi is not None else f.value, reverse=True)
        result["headline"] = f"按{primary}排序，检索到 {len(rows)}/{len(target_codes)} 家公司的可比数值。"
        if rows:
            top = rows[0]
            result["headline"] = (
                f"{top.company}的{primary}最高，为{top.money_text()}"
                + (f"（同比{_fmt_yoy(top.yoy)}）" if top.yoy is not None else "")
                + f"；本次共抽取到 {len(rows)}/{len(target_codes)} 家。"
            )
        table_rows = []
        citations = []
        for i, f in enumerate(rows, 1):
            table_rows.append([str(i), f.company, f.money_text(), _fmt_yoy(f.yoy), f"{f.page}"])
            citations.append(_citation_from_fact(f, i))
        result["table"] = {"columns": ["排名", "公司", primary, "同比", "页码"], "rows": table_rows}
        result["bullets"] = [
            f"第{i}名 {f.company}：{f.money_text()}，同比{_fmt_yoy(f.yoy)} [{i}]"
            for i, f in enumerate(rows[:5], 1)
        ]
        result["citations"] = citations
        if len(rows) < len(target_codes):
            result["answer_status"] = "partial"
            missing = [BY_CODE[c].name for c in target_codes if c not in {f.code for f in rows}]
            result["notes"].append(f"未抽取到：{'、'.join(missing)}")
        return result

    bullets = []
    citations = []
    for i, f in enumerate(facts, 1):
        yoy_txt = f"，同比{_fmt_yoy(f.yoy)}" if f.yoy is not None else ""
        bullets.append(f"{f.company} 2024年{f.metric}为{f.money_text()}{yoy_txt}，见第{f.page}页 [{i}]")
        citations.append(_citation_from_fact(f, i))
    result["headline"] = bullets[0] if bullets else "未能抽取到数值。"
    result["bullets"] = bullets
    result["citations"] = citations
    result["table"] = {
        "columns": ["公司", "指标", "2024年", "同比", "页码"],
        "rows": [[f.company, f.metric, f.money_text(), _fmt_yoy(f.yoy), str(f.page)] for f in facts],
    }
    return result


def load_retriever() -> Retriever:
    return Retriever(HybridIndex.load())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    r = load_retriever()
    print(json.dumps(answer_question(args.question, r, args.top_k), ensure_ascii=False, indent=2))
