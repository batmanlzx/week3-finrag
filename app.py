"""Flask UI for the 白酒 annual-report QA system."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401,E402

from flask import Flask, abort, jsonify, render_template, request, send_file  # noqa: E402

from src.config import BY_CODE, COMPANIES, INDEX, META, RAW_PDF  # noqa: E402
from src.qa import answer_question, load_retriever  # noqa: E402

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
retriever = None


def get_retriever():
    global retriever
    if retriever is None:
        retriever = load_retriever()
    return retriever


@app.get("/")
def index():
    summary_path = META / "extract_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    index_meta_path = INDEX / "index_meta.json"
    index_meta = json.loads(index_meta_path.read_text(encoding="utf-8")) if index_meta_path.exists() else {}
    return render_template(
        "index.html",
        companies=[{"code": c.code, "name": c.name} for c in COMPANIES],
        summary=summary,
        index_meta=index_meta,
    )


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if len(question) > 300:
        return jsonify({"error": "问题过长（最多 300 字）"}), 400
    top_k = min(max(int(payload.get("top_k", 12)), 4), 24)
    t0 = time.perf_counter()
    try:
        result = answer_question(question, get_retriever(), top_k=top_k)
    except Exception as e:  # noqa: BLE001
        app.logger.exception("ask failed")
        return jsonify({"error": f"检索失败：{e}"}), 500
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return app.response_class(
        json.dumps(result, ensure_ascii=False),
        mimetype="application/json; charset=utf-8",
    )


@app.get("/api/health")
def health():
    meta = {}
    p = INDEX / "index_meta.json"
    if p.exists():
        meta = json.loads(p.read_text(encoding="utf-8"))
    return jsonify({"ok": True, "index": meta})


@app.get("/pdf/<code>")
def pdf(code: str):
    company = BY_CODE.get(code)
    if not company:
        abort(404)
    path = RAW_PDF / f"{code}_{company.name}_2024_annual.pdf"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/pdf", as_attachment=False, download_name=path.name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
