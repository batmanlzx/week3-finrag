"""PDF -> page text + row/column tables -> metadata-rich chunks.

Pipeline:
  1. PyMuPDF extracts text per page and detects ruled tables.
  2. Section headings (第X节 ...) are tracked across pages, including headings
     split over two lines and pages that contain two sections.
  3. Prose is packed into ~800-character chunks at sentence boundaries.
  4. Every table becomes one or more pipe-delimited table chunks, with the
     header row repeated when a table is split.
Each chunk keeps company / section / page range / content type / source URL so
that every answer can be traced back to a page of the filed PDF.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401,E402

import pymupdf  # noqa: E402

from src.config import COMPANIES, META, PROCESSED, RAW_PDF, REPORT_YEAR  # noqa: E402

SECTION_RE = re.compile(r"^\s*第([一二三四五六七八九十百]+)节\s*(.*)$")
CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九", 0)}


def cn_to_int(s: str) -> int:
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = CN_NUM.get(left, 1) if left else 1
        ones = CN_NUM.get(right, 0) if right else 0
        return tens * 10 + ones
    total = 0
    for ch in s:
        if ch in CN_NUM:
            total = total * 10 + CN_NUM[ch]
    return total


def clean_page_text(text: str) -> str:
    """Strip running headers/footers but keep line boundaries for section parsing."""
    lines = []
    for raw in text.splitlines():
        line = raw.replace("\u3000", " ").strip()
        if not line:
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if "年年度报告" in line and len(line) < 40 and re.search(r"\d{6}", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _heading_at(lines: list[str], i: int) -> tuple[int, str, int] | None:
    """Parse a section heading at line i. Return (no, title, lines_consumed)."""
    m = SECTION_RE.match(lines[i])
    if not m:
        return None
    no = cn_to_int(m.group(1))
    if not (1 <= no <= 20):
        return None
    title = m.group(2).strip()
    consumed = 1
    if not title and i + 1 < len(lines):
        cand = lines[i + 1].strip()
        if 0 < len(cand) <= 40 and "..." not in cand and "…" not in cand:
            title = cand
            consumed = 2
    title = re.sub(r"[.．·…\s]+\d*$", "", title).strip(" .．·")
    if not title or len(title) > 40 or "..." in title or "…" in title:
        return None
    return no, title, consumed


def split_page_sections(
    page_text: str, current: tuple[int, str] | None
) -> tuple[list[tuple[tuple[int, str], str]], tuple[int, str] | None]:
    """Split one page into (section, text) segments, carrying section state."""
    lines = page_text.splitlines()
    segments: list[tuple[tuple[int, str], str]] = []
    sec = current or (0, "正文")
    buf: list[str] = []
    i = 0
    while i < len(lines):
        head = _heading_at(lines, i)
        if head:
            no, title, consumed = head
            if buf:
                segments.append((sec, "\n".join(buf)))
                buf = []
            sec = (no, title)
            i += consumed
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        segments.append((sec, "\n".join(buf)))
    return segments, sec


def section_heading_positions(page: pymupdf.Page) -> list[tuple[float, tuple[int, str]]]:
    """Y positions of section headings, for assigning tables to the right section."""
    out = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, txt, *_ = block
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            continue
        head = _heading_at(lines, 0)
        if head:
            out.append((y0, (head[0], head[1])))
    return out


def clean_cell(cell: str | None) -> str:
    if not cell:
        return ""
    s = cell.replace("\u3000", " ").replace("|", "／")
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def table_to_pipe(rows: list[list[str]], max_cols: int | None = None) -> list[str]:
    rows = [[clean_cell(c) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    if max_cols:
        width = min(width, max_cols)
    norm = [(r + [""] * width)[:width] for r in rows]
    lines = ["| " + " | ".join(norm[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    for r in norm[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return lines


def caption_before(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> str:
    y0 = bbox[1]
    best = ""
    for block in page.get_text("blocks"):
        x0, by0, x1, by1, txt, *_ = block
        if by1 <= y0 + 2 and by1 >= y0 - 90:
            t = clean_cell(txt)
            if t and len(t) <= 60:
                best = t
    return best


def pack_sentences(text: str, target: int = 820, overlap: int = 120) -> list[str]:
    # join soft-wrapped Chinese lines first
    text = re.sub(r"(?<=[\u4e00-\u9fff，。；：、！？）】”’])\n(?=[\u4e00-\u9fff（【“‘])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = [s for s in re.split(r"(?<=[。！？；])", text) if s.strip()]
    if not sentences:
        sentences = [text]
    chunks, buf, size = [], [], 0
    for sent in sentences:
        if size + len(sent) > target and buf:
            chunk = "".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            tail, tail_len = [], 0
            for s in reversed(buf):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            buf = tail
            size = tail_len
        buf.append(sent)
        size += len(sent)
    if buf:
        chunk = "".join(buf).strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)
    return chunks


def extract_company(company, manifest: dict) -> dict:
    rec = manifest["reports"][company.code]
    pdf_path = RAW_PDF / Path(rec["file"]).name
    doc = pymupdf.open(pdf_path)
    page_records = []
    chunks = []
    current_section: tuple[int, str] | None = None
    seq = 0

    for pno in range(doc.page_count):
        page = doc[pno]
        page_text = clean_page_text(page.get_text("text"))
        segments, current_section = split_page_sections(page_text, current_section)
        if not segments:
            segments = [(current_section or (0, "正文"), page_text)]
        heading_pos = section_heading_positions(page)
        page_first_section = segments[0][0]

        for sec, seg_text in segments:
            no, title = sec
            section = f"第{no}节 {title}" if no else title
            for piece in pack_sentences(seg_text):
                seq += 1
                header = f"{company.name} 2024年年度报告｜{section}｜第{pno + 1}页"
                chunks.append({
                    "chunk_id": f"{company.code}-p{pno + 1}-t{seq:04d}",
                    "company": company.name,
                    "code": company.code,
                    "year": REPORT_YEAR,
                    "section": section,
                    "page_start": pno + 1,
                    "page_end": pno + 1,
                    "content_type": "text",
                    "text": f"{header}\n{piece}",
                    "raw_text": piece,
                    "pdf_url": rec["pdf_url"],
                    "pdf_file": rec["file"],
                })

        page_records.append({
            "company": company.name,
            "code": company.code,
            "page": pno + 1,
            "section": f"第{page_first_section[0]}节 {page_first_section[1]}" if page_first_section[0] else page_first_section[1],
            "text": page_text,
        })

        try:
            found = page.find_tables()
        except Exception:
            found = None
        if found:
            for ti, table in enumerate(found.tables):
                try:
                    rows = table.extract()
                except Exception:
                    continue
                pipe = table_to_pipe(rows, max_cols=12)
                if len(pipe) < 3:
                    continue
                caption = caption_before(page, table.bbox)
                # pick nearest section heading above this table
                sec = page_first_section
                for y, cand in heading_pos:
                    if y <= table.bbox[1] + 2:
                        sec = cand
                no, title = sec
                section = f"第{no}节 {title}" if no else title
                header = f"{company.name} 2024年年度报告｜{section}｜第{pno + 1}页"
                lab = f"【表格】{caption}" if caption else "【表格】"
                body = pipe[2:]
                groups = [body[i:i + 24] for i in range(0, len(body), 24)] or [[]]
                for gi, group in enumerate(groups, 1):
                    seq += 1
                    table_md = "\n".join(pipe[:2] + group)
                    suffix = f"（第{gi}段）" if len(groups) > 1 else ""
                    chunks.append({
                        "chunk_id": f"{company.code}-p{pno + 1}-tab{ti + 1}-{gi}",
                        "company": company.name,
                        "code": company.code,
                        "year": REPORT_YEAR,
                        "section": section,
                        "page_start": pno + 1,
                        "page_end": pno + 1,
                        "content_type": "table",
                        "table_index": ti + 1,
                        "caption": caption,
                        "page_hint": page_text[:500],
                        "text": f"{header}\n{lab}{suffix}\n{table_md}",
                        "raw_text": table_md,
                        "pdf_url": rec["pdf_url"],
                        "pdf_file": rec["file"],
                    })

    total_pages = doc.page_count
    doc.close()

    out_pages = PROCESSED / "pages" / f"{company.code}.jsonl"
    out_pages.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pages, "w", encoding="utf-8") as f:
        for row in page_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_text = sum(1 for c in chunks if c["content_type"] == "text")
    n_tab = sum(1 for c in chunks if c["content_type"] == "table")
    print(
        f"{company.name:<6} pages={total_pages:>3} text_chunks={n_text:>4} "
        f"table_chunks={n_tab:>4} chars={sum(len(c['raw_text']) for c in chunks):>8,}"
    )
    return {
        "code": company.code,
        "name": company.name,
        "pages": total_pages,
        "text_chunks": n_text,
        "table_chunks": n_tab,
        "chars": sum(len(c["raw_text"]) for c in chunks),
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma separated ticker codes")
    args = parser.parse_args()

    manifest = json.loads((META / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(args.only.split(",")) if args.only else None
    all_chunks = []
    stats = []
    for company in COMPANIES:
        if wanted and company.code not in wanted:
            continue
        if company.code not in manifest.get("reports", {}):
            print(f"skip {company.name}: no PDF in manifest")
            continue
        stats.append(extract_company(company, manifest))
        all_chunks.extend(stats[-1]["chunks"])

    out = PROCESSED / "chunks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    summary = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "companies": len(stats),
        "pages": sum(s["pages"] for s in stats),
        "text_chunks": sum(s["text_chunks"] for s in stats),
        "table_chunks": sum(s["table_chunks"] for s in stats),
        "chars": sum(s["chars"] for s in stats),
        "per_company": [{k: v for k, v in s.items() if k != "chunks"} for s in stats],
    }
    (META / "extract_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_company"}, ensure_ascii=False, indent=2))
    print("chunks ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
