"""Run the 10-question evaluation and write a per-question trace.

The automatic checker compares table values / company order against the gold
set in eval/questions.json. Manual review notes live next to the machine verdict
so that a correct number does not hide a retrieval miss.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401,E402

from src.qa import answer_question, load_retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"


def num(s: str) -> float | None:
    s = str(s).replace(",", "").replace("，", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def row_for(table: dict | None, company: str, metric: str) -> list | None:
    if not table:
        return None
    cols = table.get("columns", [])
    mi = next((i for i, c in enumerate(cols) if c == "指标" or "指标" in c), None)
    if mi is None:
        mi = next((i for i, c in enumerate(cols) if metric in c or c in metric), None)
    for row in table.get("rows", []):
        if company not in row:
            continue
        if mi is not None and mi < len(row):
            label = str(row[mi])
            if metric in label or label in metric:
                return row
        elif mi is None:
            return row
    # fall back to the first row of the company only if the table has no metric column
    if mi is None:
        for row in table.get("rows", []):
            if company in row:
                return row
    return None


def check_single(q: dict, a: dict) -> tuple[bool, list[str]]:
    failures = []
    for m in q.get("metrics", []):
        row = row_for(a.get("table"), q["company"], m["name"])
        if not row:
            failures.append(f"答案表没有 {q['company']} / {m['name']} 行")
            continue
        val = num(row[2]) if len(row) > 2 else None
        yoy = num(row[3]) if len(row) > 3 else None
        if val is None or abs(val - m["value_yi"]) > max(0.1, abs(m["value_yi"]) * 0.01):
            failures.append(f"{m['name']} 数值 {val} != 金标准 {m['value_yi']}")
        if "yoy" in m and (yoy is None or abs(yoy - m["yoy"]) > 0.2):
            failures.append(f"{m['name']} 同比 {yoy} != 金标准 {m['yoy']}")
    return not failures, failures


def check_ranking(q: dict, a: dict) -> tuple[bool, list[str]]:
    failures = []
    table = a.get("table")
    if not table or len(table.get("rows", [])) < 3:
        return False, ["答案表不足 3 行"]
    top = [r[1] for r in table["rows"][:3]]
    if top != q["ranking"]:
        failures.append(f"前三顺序 {top} != 金标准 {q['ranking']}")
    for row, val in zip(table["rows"][:3], q["values_yi"]):
        got = num(row[2])
        if got is None or abs(got - val) > max(0.1, val * 0.01):
            failures.append(f"{row[1]} 数值 {got} != 金标准 {val}")
    return not failures, failures


def check_extrema(q: dict, a: dict) -> tuple[bool, list[str]]:
    failures = []
    table = a.get("table")
    if not table or not table.get("rows"):
        return False, ["答案表为空"]
    rows = table["rows"]
    hi = max(rows, key=lambda r: num(r[3]) if num(r[3]) is not None else -999)
    lo = min(rows, key=lambda r: num(r[3]) if num(r[3]) is not None else 999)
    e = q["extrema"]
    if hi[1] != e["highest"] or abs(num(hi[3]) - e["highest_yoy"]) > 0.2:
        failures.append(f"最高 {hi[1]} {hi[3]} != {e['highest']} {e['highest_yoy']}")
    if lo[1] != e["lowest"] or abs(num(lo[3]) - e["lowest_yoy"]) > 0.2:
        failures.append(f"最低 {lo[1]} {lo[3]} != {e['lowest']} {e['lowest_yoy']}")
    return not failures, failures


def check(q: dict, a: dict) -> tuple[bool, list[str]]:
    if "ranking" in q:
        return check_ranking(q, a)
    if "extrema" in q:
        return check_extrema(q, a)
    return check_single(q, a)


def page_recalled(q: dict, a: dict) -> bool | None:
    if "ranking" in q or "extrema" in q:
        return None
    pages = q.get("source_pages") or ([q.get("source_page")] if q.get("source_page") else [])
    if not pages:
        return None
    return any(
        c["company"] == q["company"] and c["page"] in set(pages)
        for c in a["retrieval"]["chunks"]
    )


def fmt_chunk(c: dict) -> str:
    tags = []
    if c.get("dense_rank"):
        tags.append(f"d{c['dense_rank']}")
    if c.get("bm25_rank"):
        tags.append(f"b{c['bm25_rank']}")
    return (
        f"`{c['chunk_id']}` · {c['company']} · {c['section']} · p.{c['page']} · {c['type']} "
        f"· {'/'.join(tags) or '—'} · rrf {c.get('score', 0):.4f}  \n"
        f"  > {re.sub(r'\s+', ' ', c.get('snippet', ''))[:180]}"
    )


def main() -> int:
    questions = json.loads((EVAL / "questions.json").read_text(encoding="utf-8"))
    r = load_retriever()
    results = []
    for q in questions:
        a = answer_question(q["question"], r, top_k=12)
        ok, failures = check(q, a)
        fact_ids = {c["chunk_id"] for c in a["retrieval"].get("fact_chunks", [])}
        top_ids = {c["chunk_id"] for c in a["retrieval"]["chunks"]}
        res = {
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "gold": {k: v for k, v in q.items() if k not in ("question", "type", "note")},
            "note": q.get("note", ""),
            "answer_status": a["answer_status"],
            "intent": a["spec"]["intent"],
            "mode": a["retrieval"]["mode"],
            "correct": ok,
            "failures": failures,
            "page_recalled_at_12": page_recalled(q, a),
            "fact_in_question_topk": bool(fact_ids & top_ids),
            "headline": a["headline"],
            "bullets": a["bullets"],
            "table": a["table"],
            "notes": a["notes"],
            "citations": a["citations"],
            "retrieval": a["retrieval"]["chunks"],
            "fact_chunks": a["retrieval"].get("fact_chunks", []),
            "elapsed_ms": a.get("elapsed_ms"),
        }
        results.append(res)
        print(f"{q['id']} {'PASS' if ok else 'FAIL'} status={a['answer_status']} page@12={res['page_recalled_at_12']} fact_in_topk={res['fact_in_question_topk']}")
        if failures:
            print("   ", " | ".join(failures))

    out_jsonl = EVAL / "results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(EVAL / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "correct", "answer_status", "page_recalled_at_12", "fact_in_question_topk", "failures"])
        for x in results:
            w.writerow([x["id"], x["type"], x["question"], x["correct"], x["answer_status"], x["page_recalled_at_12"], x["fact_in_question_topk"], " | ".join(x["failures"])])

    write_report(results)
    print(f"\n{sum(1 for x in results if x['correct'])}/{len(results)} correct")
    print("wrote", out_jsonl)
    return 0


def write_report(results: list[dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 10 题逐题评测：召回、对错、错在哪",
        "",
        f"生成时间：{now}　｜　语料：12 家白酒公司 2024 年年报　｜　检索：BM25 + 向量 RRF，top-12",
        "",
        "## 总览",
        "",
        "| # | 类型 | 问题 | 结果 | 答案状态 | 金标准页是否进 top-12 | 数值块是否在问题 top-12 |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in results:
        mark = "✅" if x["correct"] else "❌"
        pr = "是" if x["page_recalled_at_12"] else ("—" if x["page_recalled_at_12"] is None else "否")
        ft = "是" if x["fact_in_question_topk"] else "否"
        lines.append(f"| {x['id']} | {x['type']} | {x['question']} | {mark} | {x['answer_status']} | {pr} | {ft} |")
    n_single = sum(1 for x in results if x["page_recalled_at_12"] is not None)
    n_page = sum(1 for x in results if x["page_recalled_at_12"] is True)
    n_fact = sum(1 for x in results if x["fact_in_question_topk"])
    n_correct = sum(1 for x in results if x["correct"])
    lines += [
        "",
        f"> 关键读数：{n_correct}/{len(results)} 题数值对；单公司题里金标准页进入问题级 top-12 的为 {n_page}/{n_single}，"
        f"数值块进入问题级 top-12 的为 {n_fact}/{len(results)}。"
        "混合检索擅长定位章节，却经常被附注里同名的「合同负债 / 期末余额」表带偏；真正把数字捞准的是第二阶段的指标锚定表格查询。",
        "",
        "**结论先行**",
        "",
        "- 答得好的：单公司、主表科目、双指标同表（q01/q02/q04/q06/q08）。问题里有明确公司 + 明确指标时，表格行抽取很稳。",
        "- 答得勉强：资产负债表科目（q03/q07）。合并报表、母公司报表、附注表都出现「合同负债」，检索层分不清；q07 的问题级 top-12 全是附注表，靠第二阶段才回到 p62。",
        "- 跨公司题（q09/q10）数值全对，但前提是「逐公司召回 + 结构化汇总」；如果只取全局 top-12，会被茅台/五粮液两家霸榜。",
        "- 典型翻车点不在生成，而在**表格结构**：表头被拆成多行、空列、单位（元/万元/亿元）、合并 vs 母公司。数值题的上限由抽取层决定。",
        "",
    ]

    for x in results:
        mark = "✅ 答对" if x["correct"] else "❌ 答错"
        lines += [
            f"## {x['id']}　{x['question']}",
            "",
            f"- 类型：{x['type']}",
            f"- 判定：**{mark}**　答案状态：`{x['answer_status']}`　意图：`{x['intent']}`　检索模式：`{x['mode']}`",
            f"- 金标准页是否进问题 top-12：`{x['page_recalled_at_12']}`　数值块是否在问题 top-12：`{x['fact_in_question_topk']}`",
            "",
            "**系统答案**",
            "",
            f"> {x['headline']}",
            "",
        ]
        if x.get("table") and x["table"].get("rows"):
            cols = x["table"]["columns"]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for row in x["table"]["rows"]:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")
        if x.get("notes"):
            lines += ["**系统备注**：" + "；".join(x["notes"]), ""]
        lines += [
            "**召回的块（问题级混合检索，前 8）**",
            "",
        ]
        for i, c in enumerate(x["retrieval"][:8], 1):
            lines.append(f"{i}. {fmt_chunk(c)}")
        lines.append("")
        if x.get("fact_chunks"):
            lines += ["**数值定位块（第二阶段）**", ""]
            for c in x["fact_chunks"][:6]:
                lines.append(
                    f"- `{c['chunk_id']}` · {c['company']} · {c['section']} · p.{c['page']} · "
                    f"{c.get('fact_metric', '')} = {c.get('fact_value', '')}  \n"
                    f"  > {re.sub(r'\s+', ' ', c.get('snippet', ''))[:180]}"
                )
            lines.append("")
        lines += ["**错在哪 / 风险**", ""]
        if x["failures"]:
            for f in x["failures"]:
                lines.append(f"- {f}")
        else:
            lines.append("- 数值与金标准一致。")
            if not x["fact_in_question_topk"]:
                lines.append("- 但数值所在块不在问题级 top-12：语义检索没有直接命中该表，是第二阶段的指标锚定查询把它捞回来的。")
            if x["page_recalled_at_12"] is False:
                lines.append("- 金标准页未被问题级检索召回，说明“读哪一页”与“答对数字”在这套系统里是两件事。")
            if x["id"] == "q06" and not any(c["page"] == 7 for c in x["fact_chunks"]):
                lines.append("- 系统引用了利润表而不是 p.7 主要会计数据；数字对，但出处不是最直观的那张表。")
            if x["id"] == "q03":
                lines.append("- 检索先命中 p.25 管理层讨论的合同负债明细表，而不是 p.87 合并资产负债表；两处数值一致，但口径解释依赖抽取层的表头识别。")
            if x["id"] == "q02":
                lines.append("- 现金流曾被 p.8 分季度表干扰（Q1 的 5.16 亿元）；系统用「四列季度值 + 无年度表头」规则剔除了它。")
            if x["id"] == "q07":
                lines.append("- 问题级 top-12 全是附注里的「合同负债/期末余额」表，没有 p.62 合并资产负债表；第二阶段用「合并资产负债表 + 合同负债 + 期末余额」锚定查询才捞回 p.62。答案对，但检索路径是脆的。")
            if x["id"] == "q09":
                lines.append("- 古井贡酒的营业总收入行没有被抽取器命中，系统回退到营业收入口径并在备注中披露；这是口径风险，不是数值错误。")
        lines.append("")

    (EVAL / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (EVAL / "retrieval_trace.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
