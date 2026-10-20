"""Shared paths and company registry for the 白酒 annual-report RAG."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_PDF = DATA / "raw_pdf"
META = DATA / "meta"
PROCESSED = DATA / "processed"
INDEX = DATA / "index"
EVAL = ROOT / "eval"
DOCS = ROOT / "docs"

for d in (RAW_PDF, META, PROCESSED, INDEX, EVAL, DOCS):
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Company:
    code: str
    name: str
    short: str
    exchange: str
    aliases: tuple[str, ...]

    @property
    def text_aliases(self) -> list[str]:
        vals = {self.name, self.short, *self.aliases}
        return sorted(vals, key=len, reverse=True)


def load_companies() -> list[Company]:
    payload = json.loads((ROOT / "config" / "companies.json").read_text(encoding="utf-8"))
    return [
        Company(
            code=c["code"],
            name=c["name"],
            short=c["short"],
            exchange=c["exchange"],
            aliases=tuple(c["aliases"]),
        )
        for c in payload["companies"]
    ]


COMPANIES = load_companies()
BY_CODE = {c.code: c for c in COMPANIES}
BY_NAME = {c.name: c for c in COMPANIES}
REPORT_YEAR = 2024
INDUSTRY = "白酒（A股）"

# Metric synonyms used by both query expansion and answer extraction.
METRIC_SYNONYMS = {
    "营业总收入": ["营业总收入", "总营收"],
    "营业收入": ["营业收入", "营收", "收入"],
    "归母净利润": ["归属于上市公司股东的净利润", "归母净利润", "净利润", "股东净利润"],
    "扣非净利润": ["扣除非经常性损益的净利润", "扣非净利润", "扣非"],
    "经营活动现金流": ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"],
    "毛利率": ["毛利率", "毛利"],
    "研发投入": ["研发投入", "研发费用", "研发投入合计"],
    "合同负债": ["合同负债"],
    "销售费用": ["销售费用"],
    "资产负债率": ["资产负债率"],
    "基本每股收益": ["基本每股收益", "每股收益"],
}
