(() => {
  const $ = (sel) => document.querySelector(sel);
  const form = $("#ask-form");
  const input = $("#question");
  const btn = $("#ask-btn");
  const loading = $("#loading");
  const answer = $("#answer");
  const badges = $("#badges");
  const time = $("#time");
  const headline = $("#headline");
  const spec = $("#spec");
  const tableWrap = $("#table-wrap");
  const factPath = $("#fact-path");
  const bullets = $("#bullets");
  const notes = $("#notes");
  const evidence = $("#evidence");
  const evCount = $("#ev-count");

  let current = null;
  let evLimit = 8;

  const esc = (s) => String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const citeLink = (n, map) => {
    const c = map.get(Number(n));
    if (!c) return `<span class="cite">${esc(n)}</span>`;
    return `<a class="cite" href="${esc(c.pdf_url || "#")}" target="_blank" rel="noopener" title="${esc(c.company)} · 第${esc(c.page)}页">${esc(n)}</a>`;
  };

  const withCites = (text, map) => esc(text).replace(/\[(\d+)\]/g, (_, n) => citeLink(n, map));

  function statusInfo(status) {
    if (status === "ok") return ["ok", "已核验"];
    if (status === "partial") return ["partial", "部分覆盖"];
    return ["failed", "未命中"];
  }

  function renderTable(table, citations) {
    const map = new Map(citations.map(c => [c.n, c]));
    let html = "<table><thead><tr>";
    for (const col of table.columns) html += `<th>${esc(col)}</th>`;
    html += "</tr></thead><tbody>";
    table.rows.forEach((row, ri) => {
      html += "<tr>";
      row.forEach((cell, ci) => {
        if (ci === 4 && citations[ri]) {
          const c = citations[ri];
          html += `<td><a href="${esc(c.pdf_url)}" target="_blank" rel="noopener" style="color:var(--accent);text-decoration:none;border-bottom:1px solid #e3c0b6">p.${esc(cell)}</a></td>`;
        } else {
          html += `<td>${esc(cell)}</td>`;
        }
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    return html;
  }

  function renderEvidence(items, limit) {
    const shown = items.slice(0, limit);
    evidence.innerHTML = shown.map((ev, i) => {
      const rank = ev.dense_rank ? `d${ev.dense_rank}` : "";
      const brank = ev.bm25_rank ? `b${ev.bm25_rank}` : "";
      return `<article class="ev">
        <div class="ev-rank">#${i + 1}<b>${ev.type === "table" ? "T" : "P"}</b></div>
        <div class="ev-main">
          <div class="ev-top">
            <span class="ev-company">${esc(ev.company)}</span>
            <span class="ev-tag ${ev.type === "table" ? "table" : ""}">${ev.type === "table" ? "表格" : "正文"}</span>
            <span class="ev-section">${esc(ev.section)} · p.${esc(ev.page)}</span>
          </div>
          <div class="ev-snippet">${esc(ev.snippet)}</div>
        </div>
        <div class="ev-side">
          <div>${rank} ${brank}</div>
          <div>rrf ${Number(ev.score).toFixed(4)}</div>
          <a href="/pdf/${esc(ev.code)}#page=${esc(ev.page)}" target="_blank" rel="noopener">打开原文 ↗</a>
        </div>
      </article>`;
    }).join("");
    evCount.textContent = `${items.length} 块 · 展示 ${Math.min(limit, items.length)}`;
  }

  function render(data) {
    current = data;
    evLimit = 8;
    answer.classList.remove("hidden");
    const [cls, label] = statusInfo(data.answer_status);
    const modeLabel = data.retrieval.mode === "panorama" ? "逐公司召回" : (data.retrieval.mode === "single" ? "公司定向检索" : "全库检索");
    badges.innerHTML = `<span class="badge ${cls}">${label}</span>
      <span class="badge mode">${modeLabel}</span>
      <span class="badge">${esc(data.spec.intent)}</span>
      <span class="badge">${data.retrieval.chunks.length} 块证据</span>`;
    time.textContent = `${data.elapsed_ms} ms · top-k ${data.retrieval.chunks.length}`;
    headline.textContent = data.headline || "";

    const specBits = [];
    if (data.spec.companies.length) specBits.push(`<span>公司 <b>${esc(data.spec.companies.join("、"))}</b></span>`);
    if (data.spec.metrics.length) specBits.push(`<span>指标 <b>${esc(data.spec.metrics.join("、"))}</b></span>`);
    specBits.push(`<span>年份 <b>${esc(data.spec.years.join("/"))}</b></span>`);
    spec.innerHTML = specBits.join("");

    const citations = data.citations || [];
    const map = new Map(citations.map(c => [c.n, c]));

    if (data.table && data.table.rows && data.table.rows.length) {
      tableWrap.innerHTML = renderTable(data.table, citations);
      tableWrap.classList.remove("hidden");
    } else {
      tableWrap.classList.add("hidden");
      tableWrap.innerHTML = "";
    }

    const factChunks = (data.retrieval.fact_chunks || []);
    if (factChunks.length) {
      factPath.innerHTML = `<span class="fact-path-label">数值定位</span>` + factChunks.slice(0, 6).map(f =>
        `<a class="fact-chip" href="/pdf/${esc(f.code)}#page=${esc(f.page)}" target="_blank" rel="noopener" title="${esc(f.section)} · ${esc(f.chunk_id)}">
          ${esc(f.company)} · ${esc(f.fact_metric)} <b>${esc(f.fact_value)}</b> · p.${esc(f.page)} ↗
        </a>`).join("");
      factPath.classList.remove("hidden");
    } else {
      factPath.classList.add("hidden");
      factPath.innerHTML = "";
    }

    bullets.innerHTML = (data.bullets || []).map(b => `<div class="bullet">${withCites(b, map)}</div>`).join("");

    if (data.notes && data.notes.length) {
      notes.innerHTML = data.notes.map(n => esc(n)).join(" · ");
      notes.classList.remove("hidden");
    } else {
      notes.classList.add("hidden");
    }

    renderEvidence(data.retrieval.chunks || [], evLimit);
    document.querySelectorAll(".ev-more").forEach(el => el.remove());

    if (data.retrieval.chunks.length > evLimit) {
      const more = document.createElement("div");
      more.className = "ev-more";
      more.innerHTML = `<button type="button">展开全部 ${data.retrieval.chunks.length} 块证据</button>`;
      more.querySelector("button").addEventListener("click", () => {
        evLimit = data.retrieval.chunks.length;
        renderEvidence(data.retrieval.chunks, evLimit);
        more.remove();
      });
      evidence.after(more);
    }
  }

  async function ask(question) {
    const q = (question || input.value || "").trim();
    if (!q) return;
    loading.classList.remove("hidden");
    answer.classList.add("hidden");
    btn.disabled = true;
    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: 12 })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "请求失败");
      render(data);
    } catch (err) {
      answer.classList.remove("hidden");
      badges.innerHTML = `<span class="badge failed">请求失败</span>`;
      headline.textContent = String(err.message || err);
      spec.innerHTML = "";
      bullets.innerHTML = "";
      tableWrap.classList.add("hidden");
      notes.classList.add("hidden");
      evidence.innerHTML = "";
      evCount.textContent = "";
    } finally {
      loading.classList.add("hidden");
      btn.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); ask(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
  document.querySelectorAll(".chips button").forEach(b => {
    b.addEventListener("click", () => { input.value = b.dataset.q; ask(b.dataset.q); });
  });

  const params = new URLSearchParams(location.search);
  const initial = params.get("q");
  if (initial) input.value = initial;
  if (input.value.trim()) setTimeout(() => ask(input.value), 250);
})();
