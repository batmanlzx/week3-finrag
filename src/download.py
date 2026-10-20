"""Download 2024 annual reports from CNINFO (巨潮资讯).

The disclosure API is public but undocumented; we query the full-text
announcement endpoint, filter for the full annual report (not the summary or
English version), then fetch the PDF from static.cninfo.com.cn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401,E402

from src.config import BY_CODE, COMPANIES, META, RAW_PDF, REPORT_YEAR  # noqa: E402

BASE = "https://www.cninfo.com.cn"
STATIC = "https://static.cninfo.com.cn/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CTX = ssl.create_default_context()
BAD_TITLE = ("摘要", "英文", "问询", "回复", "审计", "关于", "公告", "更正", "取消", "说明", "意见", "核查")


def http_json(url: str, data: dict | None = None, timeout: int = 40):
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": BASE + "/new/commonUrl?url=disclosure/list/notice",
        "Origin": BASE,
    }
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
        headers["X-Requested-With"] = "XMLHttpRequest"
    req = urllib.request.Request(url, data=body, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def stock_registry() -> dict[str, dict]:
    payload = http_json(BASE + "/new/data/szse_stock.json")
    return {s["code"]: s for s in payload["stockList"]}


def query_reports(code: str, org_id: str) -> list[dict]:
    params = {
        "pageNum": "1",
        "pageSize": "30",
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},{org_id}",
        "searchkey": "年度报告",
        "secDate": "",
        "tdate": "",
        "type": "category_ndbg_szsh",
        "trade": "",
        "seDate": f"{REPORT_YEAR + 1}-01-01~{REPORT_YEAR + 1}-12-31",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "false",
    }
    payload = http_json(BASE + "/new/hisAnnouncement/query", params)
    return payload.get("announcements") or []


def pick_report(anns: list[dict], company_name: str) -> dict | None:
    scored = []
    for a in anns:
        title = (a.get("announcementTitle") or "").strip()
        if f"{REPORT_YEAR}年年度报告" not in title:
            continue
        if any(word in title for word in BAD_TITLE):
            continue
        score = 0
        if title.endswith(f"{REPORT_YEAR}年年度报告"):
            score += 10
        if company_name in title:
            score += 5
        score -= len(title) / 100
        scored.append((score, a))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def download(url: str, dest: Path, expected_kb: int | None = None) -> tuple[int, str]:
    if dest.exists() and dest.stat().st_size > 100_000:
        data = dest.read_bytes()
        if data[:4] == b"%PDF":
            return len(data), hashlib.sha256(data).hexdigest()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Referer": BASE + "/",
            "Accept": "application/pdf,*/*",
        },
    )
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                raise RuntimeError(f"not a PDF, first bytes={data[:16]!r}")
            if expected_kb and len(data) < expected_kb * 300:
                raise RuntimeError(f"truncated? got {len(data)} bytes, expected ~{expected_kb} KB")
            dest.write_bytes(data)
            return len(data), hashlib.sha256(data).hexdigest()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload even if file exists")
    args = parser.parse_args()

    manifest_path = META / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"reports": {}}
    registry = stock_registry()
    print(f"CNINFO stock registry: {len(registry)} tickers")

    failures = []
    for i, company in enumerate(COMPANIES, 1):
        row = registry.get(company.code)
        if not row:
            failures.append((company.code, "not in CNINFO registry"))
            print(f"[{i:02d}/{len(COMPANIES)}] {company.name}: NOT FOUND")
            continue
        existing = manifest["reports"].get(company.code)
        dest = RAW_PDF / f"{company.code}_{company.name}_{REPORT_YEAR}_annual.pdf"
        if existing and dest.exists() and not args.force:
            print(f"[{i:02d}/{len(COMPANIES)}] {company.name}: cached {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
            continue

        try:
            anns = query_reports(company.code, row["orgId"])
            picked = pick_report(anns, company.name)
            if not picked:
                raise RuntimeError(f"no full report among {len(anns)} announcements")
            title = picked["announcementTitle"].strip()
            rel = picked["adjunctUrl"].lstrip("/")
            url = rel if rel.startswith("http") else STATIC + rel
            size, sha = download(url, dest, picked.get("adjunctSize"))
            manifest["reports"][company.code] = {
                "code": company.code,
                "name": company.name,
                "title": title,
                "announcement_url": BASE + "/new/announcement/download?bulletinId=" + str(picked.get("announcementId", "")),
                "pdf_url": url,
                "file": str(dest.relative_to(META.parents[0])).replace("\\", "/"),
                "bytes": size,
                "sha256": sha,
                "announcement_time": datetime.fromtimestamp(
                    (picked.get("announcementTime") or 0) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i:02d}/{len(COMPANIES)}] {company.name}: {title} -> {dest.name} ({size/1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            failures.append((company.code, str(e)))
            print(f"[{i:02d}/{len(COMPANIES)}] {company.name}: FAILED {e}")

    manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["count"] = len(manifest.get("reports", {}))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDownloaded {manifest['count']}/{len(COMPANIES)} reports -> {RAW_PDF}")
    if failures:
        print("Failures:")
        for code, msg in failures:
            print(" ", code, msg)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
