const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function getApiBase() {
  const saved = localStorage.getItem("cdc_api_base") || "";
  const defaultBase = `http://${window.location.hostname}:8000`;
  const el = $("apiBase");
  const input = (el ? el.value.trim() : "") || saved.trim() || defaultBase;
  localStorage.setItem("cdc_api_base", input);
  return input.replace(/\/+$/, "");
}

function setBusy(busy) {
  if ($("btnRun")) $("btnRun").disabled = busy;
  if ($("btnStop")) $("btnStop").disabled = !busy;
  if ($("btnReload")) $("btnReload").disabled = !busy;
}

function clearRunUI() {
  if ($("runErr")) $("runErr").textContent = "";
  if ($("statusErr")) $("statusErr").textContent = "";
  if ($("plan_id")) $("plan_id").textContent = "-";
  if ($("status")) $("status").textContent = "-";
  if ($("output_path")) $("output_path").textContent = "-";
  if ($("plan_text")) $("plan_text").textContent = "";
  if ($("plan_json")) $("plan_json").textContent = "";
  if ($("download")) $("download").style.display = "none";
  if ($("btnFetchPlanJson")) $("btnFetchPlanJson").disabled = true;
}

function buildPayload() {
  const payload = {
    disease_type: ($("disease_type")?.value || "").trim() || "covid19",
    location: ($("location")?.value || "").trim() || "未提供",
    population: Number($("population")?.value || 3000),
    reported_cases: Number($("reported_cases")?.value || 25),
    underreport_factor: Number($("underreport_factor")?.value || 1.5),
    days: Number($("days")?.value || 7),
    jurisdiction: ($("jurisdiction")?.value || "").trim() || "某市疾控中心",
  };
  const regionProfile = ($("region_profile")?.value || "").trim();
  if (regionProfile) payload.region_profile = regionProfile;
  const fmt = ($("output_format")?.value || "").trim();
  if (fmt) payload.output_format = fmt;
  const out = ($("output_docx")?.value || "").trim();
  if (out) payload.output_path = out;
  return payload;
}

async function apiFetch(path, init) {
  const base = getApiBase();
  const url = `${base}${path}`;
  const resp = await fetch(url, init);
  const text = await resp.text();
  let json;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }
  if (!resp.ok) {
    const detail = (json && (json.detail || json.message)) || text || `${resp.status}`;
    throw new Error(detail);
  }
  return json;
}

let currentPlanId = null;
let timer = null;
let currentPage = document.querySelector("[data-page]")?.getAttribute("data-page") || "dashboard";

function stopPolling() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  setBusy(false);
}

function showManualFix(show) {
  if ($("manualFixErr")) $("manualFixErr").textContent = "";
  if ($("manualFixCard")) $("manualFixCard").style.display = show ? "block" : "none";
}

async function refreshManualFixUI() {
  if (!currentPlanId) return;
  try {
    const j = await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/manual_fix`);
    if (!j.pending) {
      showManualFix(false);
      return;
    }
    showManualFix(true);
    if ($("manualFixHint")) $("manualFixHint").textContent = j.manual_fix_path ? `文件路径：${j.manual_fix_path}` : "";
    const plan = await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/plan.json`).catch(() => null);
    if (plan && $("manualFixText")) $("manualFixText").value = JSON.stringify(plan, null, 2);
  } catch (e) {
    showManualFix(false);
  }
}

async function submitManualFix(action, planObj) {
  if (!currentPlanId) return;
  if ($("manualFixErr")) $("manualFixErr").textContent = "";
  try {
    const payload = { action };
    if (planObj) payload.plan = planObj;
    await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/manual_fix`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showManualFix(false);
    await pollOnce();
  } catch (e) {
    if ($("manualFixErr")) $("manualFixErr").textContent = String(e && e.message ? e.message : e);
  }
}

function showReview(show) {
  if ($("reviewErr")) $("reviewErr").textContent = "";
  if ($("reviewCard")) $("reviewCard").style.display = show ? "block" : "none";
}

async function refreshReviewUI() {
  if (!currentPlanId) return;
  try {
    const j = await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/review`);
    if (!j.pending) {
      showReview(false);
      return;
    }
    showReview(true);
    if ($("draftHint")) $("draftHint").textContent = j.draft_path ? `草稿路径：${j.draft_path}` : "";
    const draft = await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/draft.json`).catch(() => null);
    if (draft && $("draftJson")) $("draftJson").value = JSON.stringify(draft, null, 2);
    if (draft) renderDraftSummary(draft);
  } catch {
    showReview(false);
  }
}

async function submitReview(action, planObj) {
  if (!currentPlanId) return;
  if ($("reviewErr")) $("reviewErr").textContent = "";
  try {
    const payload = { action };
    if (planObj) payload.plan = planObj;
    await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showReview(false);
    await pollOnce();
  } catch (e) {
    if ($("reviewErr")) $("reviewErr").textContent = String(e && e.message ? e.message : e);
  }
}

function renderDraftSummary(draft) {
  if (!$("draftMeta")) return;
  const meta = draft.meta || {};
  const input = draft.input || {};
  const risk = draft.risk || {};
  const measures = Array.isArray(draft.measures) ? draft.measures : [];
  const resources = (draft.resources && Array.isArray(draft.resources.items)) ? draft.resources.items : [];
  const lines = [];
  if (meta.title) lines.push(`标题：${meta.title}`);
  if (input.event_type) lines.push(`疾病类型：${input.event_type}`);
  if (input.location) lines.push(`发生地点：${input.location}`);
  if (risk.level) lines.push(`风险等级：${risk.level}`);
  lines.push(`措施条数：${measures.length}`);
  lines.push(`物资条数：${resources.length}`);
  $("draftMeta").textContent = lines.join(" · ");
  if ($("draftDoc")) $("draftDoc").innerHTML = renderDraftDocument(draft);
  if ($("draftRaw")) $("draftRaw").textContent = JSON.stringify(draft, null, 2);
}

function renderDraftDocument(draft) {
  const meta = draft.meta || {};
  const input = draft.input || {};
  const risk = draft.risk || {};
  const measures = Array.isArray(draft.measures) ? draft.measures : [];
  const resources = (draft.resources && Array.isArray(draft.resources.items)) ? draft.resources.items : [];
  const sections = Array.isArray(draft.sections) ? draft.sections : [];

  const title = meta.title || "应急处置预案（草稿）";
  const org = meta.jurisdiction || input.jurisdiction || "疾控中心";
  const created = meta.created_at || "";

  const core = measures.filter((m) => (m.level || "").toLowerCase() === "core");
  const supp = measures.filter((m) => (m.level || "").toLowerCase() !== "core");

  const noteSec = sections.find((s) => s && s.title === "人工修改说明");
  const noteText = noteSec && Array.isArray(noteSec.paragraphs) ? noteSec.paragraphs.join(" ") : "";

  const normalizeEnum = (v) => {
    const s = String(v ?? "");
    if (!s) return "";
    if (s.includes(".")) return s.split(".").slice(-1)[0];
    return s;
  };

  const stripLeadingIndex = (t) => {
    const s = String(t ?? "").trim();
    return s.replace(/^(第)?[一二三四五六七八九十]+[、.]\s*/, "").replace(/^\d+[、.]\s*/, "");
  };

  const renderFreeSection = (sec, idx, level) => {
    const headingTag = level === 0 ? "h3" : "h4";
    const titleText = level === 0 ? `${idx}. ${stripLeadingIndex(sec.title || "")}` : stripLeadingIndex(sec.title || "");
    const paras = Array.isArray(sec.paragraphs) ? sec.paragraphs : [];
    const subs = Array.isArray(sec.subsections) ? sec.subsections : [];
    const pHtml = paras.map((p) => `<div class="para">${escapeHtml(p)}</div>`).join("");
    const subHtml = subs
      .map((sub) => `
        <h4>${escapeHtml(stripLeadingIndex(sub.title || ""))}</h4>
        ${(Array.isArray(sub.paragraphs) ? sub.paragraphs : []).map((p) => `<div class="para">${escapeHtml(p)}</div>`).join("")}
      `)
      .join("");
    return `<${headingTag}>${escapeHtml(titleText)}</${headingTag}>${pHtml}${subHtml}`;
  };

  const citeNo = new Map();
  const citeList = [];
  const citeMeasures = new Map();
  const citeKey = (c) => `${String(c?.source_file || "")}::${String(c?.excerpt || "")}`;

  for (const m of measures) {
    const cites = Array.isArray(m?.citations) ? m.citations : [];
    for (const c of cites) {
      const key = citeKey(c);
      if (!citeNo.has(key)) {
        citeNo.set(key, citeList.length + 1);
        citeList.push({ ...c, _key: key });
      }
      if (!citeMeasures.has(key)) citeMeasures.set(key, new Set());
      citeMeasures.get(key).add(String(m?.title || "未命名措施"));
    }
  }

  function measureHtml(m, idx) {
    const tag = (m.level || "").toLowerCase() === "core" ? `<span class="tag-core">核心</span>` : `<span class="tag-supp">补充</span>`;
    const cites = Array.isArray(m.citations) ? m.citations : [];
    const citeNums = cites
      .map((c) => citeNo.get(citeKey(c)))
      .filter((n) => typeof n === "number");
    const citeHtml = citeNums.length
      ? `<div class="para"><span class="muted">引用：${citeNums.map((n) => `[${n}]`).join(" ")}</span></div>`
      : `<div class="para"><span class="muted">引用：无</span></div>`;
    return `
      <div class="para"><b>${idx + 1}. ${escapeHtml(m.title || "")}</b> ${tag}</div>
      <div class="para">${escapeHtml(m.content || "")}</div>
      ${citeHtml}
    `;
  }

  const resourceRows = resources.length
    ? resources
      .map((it, i) => `<tr><td>${i + 1}</td><td>${escapeHtml(it.name || "")}</td><td>${escapeHtml(it.quantity ?? "")}</td><td>${escapeHtml(it.unit || "")}</td></tr>`)
      .join("")
    : `<tr><td colspan="4">暂无资源库存数据。</td></tr>`;

  const appendix = citeList.length
    ? `
      <h3>附：规范依据摘录（引用）</h3>
      <ol class="list">
        ${citeList
      .map((c) => {
        const key = c._key;
        const ms = citeMeasures.get(key) ? Array.from(citeMeasures.get(key)) : [];
        const msText = ms.length ? `措施：${ms.join("、")}` : "措施：-";
        return `
              <li>
                <div class="para"><b>${escapeHtml(msText)}</b></div>
                <div class="para"><span class="muted">来源：</span>${escapeHtml(c.source_file || "")}</div>
                <div class="para">${escapeHtml(c.excerpt || "")}</div>
              </li>
            `;
      })
      .join("")}
      </ol>
    `
    : "";

  const cnNum = (n) => {
    const m = {
      1: "一",
      2: "二",
      3: "三",
      4: "四",
      5: "五",
      6: "六",
      7: "七",
      8: "八",
      9: "九",
      10: "十",
      11: "十一",
      12: "十二",
      13: "十三",
      14: "十四",
      15: "十五",
      16: "十六",
      17: "十七",
      18: "十八",
      19: "十九",
      20: "二十",
    };
    return m[n] || String(n);
  };

  const sectionByTitle = new Map();
  for (const s of sections) {
    if (!s || !s.title) continue;
    const t = stripLeadingIndex(s.title);
    if (!t) continue;
    if (!sectionByTitle.has(t)) sectionByTitle.set(t, s);
  }

  const usedTitles = new Set(["人工修改说明"]);
  const outlineOrder = [
    "总则",
    "事件概况",
    "风险评估",
    "应急组织指挥体系",
    "监测、预警与报告",
    "应急响应",
    "防控措施",
    "资源与物资保障",
    "资源调配与缺口",
    "后期处置",
    "保障措施",
    "附则",
    "规范依据（章节来源）",
  ];

  let idx = 1;
  const parts = [];

  const pushSectionFromSec = (secTitle, sec) => {
    if (!sec) return;
    usedTitles.add(secTitle);
    parts.push(`<h3>${cnNum(idx)}、${escapeHtml(secTitle)}</h3>`);
    idx += 1;
    const paras = Array.isArray(sec.paragraphs) ? sec.paragraphs : [];
    const subs = Array.isArray(sec.subsections) ? sec.subsections : [];
    for (const p of paras) parts.push(`<div class="para">${escapeHtml(p)}</div>`);
    for (const sub of subs) {
      if (!sub) continue;
      parts.push(`<h4>${escapeHtml(stripLeadingIndex(sub.title || ""))}</h4>`);
      for (const p of (Array.isArray(sub.paragraphs) ? sub.paragraphs : [])) {
        parts.push(`<div class="para">${escapeHtml(p)}</div>`);
      }
    }
  };

  const pushEventOverview = () => {
    parts.push(`<h3>${cnNum(idx)}、事件概况</h3>`);
    idx += 1;
    parts.push(`<ul class="list">
      <li>疾病类型：${escapeHtml(normalizeEnum(input.event_type || ""))}</li>
      <li>发生地点：${escapeHtml(input.location || "")}</li>
      <li>区域人口：${escapeHtml(input.population ?? "")}</li>
      <li>报告病例数：${escapeHtml(input.reported_cases ?? "")}</li>
    </ul>`);
  };

  const pushRisk = () => {
    parts.push(`<h3>${cnNum(idx)}、风险评估</h3>`);
    idx += 1;
    parts.push(`<ul class="list">
      <li>风险等级：${escapeHtml(normalizeEnum(risk.level || ""))}</li>
      <li>评估结论：${escapeHtml(risk.summary || "")}</li>
      ${risk.predicted_cases_7d != null ? `<li>未来 7 天病例预测：${escapeHtml(risk.predicted_cases_7d)}</li>` : ""}
    </ul>`);
  };

  const pushMeasures = () => {
    parts.push(`<h3>${cnNum(idx)}、防控措施</h3>`);
    idx += 1;
    parts.push(
      core.length
        ? `<div class="para"><b>（一）核心措施（强合规）</b></div>${core.map(measureHtml).join("")}`
        : `<div class="para">暂无核心措施。</div>`
    );
    if (supp.length) {
      const rp = String(input.region_profile || "").trim();
      const rpText = rp ? rp.replace(/[。！？；;]+$/, "") : "";
      parts.push(
        `<div class="para" style="margin-top:10px;"><b>（二）补充措施（区域适配）</b></div>${
          rpText ? `<div class="para">适配因素：${escapeHtml(rpText)}。</div>` : ""
        }${supp.map(measureHtml).join("")}`
      );
    }
  };

  const pushResources = () => {
    parts.push(`<h3>${cnNum(idx)}、资源与物资保障</h3>`);
    idx += 1;
    parts.push(`<table class="table">
      <thead><tr><th style="width:60px;">序号</th><th>物资名称</th><th style="width:120px;">数量</th><th style="width:90px;">单位</th></tr></thead>
      <tbody>${resourceRows}</tbody>
    </table>`);
  };

  for (const t of outlineOrder) {
    if (t === "事件概况") {
      pushEventOverview();
      continue;
    }
    if (t === "风险评估") {
      pushRisk();
      continue;
    }
    if (t === "防控措施") {
      pushMeasures();
      continue;
    }
    if (t === "资源与物资保障") {
      pushResources();
      continue;
    }
    const sec = sectionByTitle.get(t);
    if (sec) pushSectionFromSec(t, sec);
  }

  for (const [t, sec] of sectionByTitle.entries()) {
    if (usedTitles.has(t)) continue;
    if (t === "人工修改说明") continue;
    pushSectionFromSec(t, sec);
  }

  parts.push(`<h3>${cnNum(idx)}、审批与签字</h3>`);
  idx += 1;
  parts.push(`<ul class="list">
      <li>拟稿：</li>
      <li>审核：</li>
      <li>批准：</li>
      <li>日期：${escapeHtml(created || "")}</li>
      <li>（公章占位）</li>
    </ul>`);

  if (appendix) parts.push(appendix);

  return `
    <div class="doc-title">${escapeHtml(title)}</div>
    <div class="doc-meta">
      <span>编制单位：${escapeHtml(org)}</span>
      ${created ? `<span>生成时间：${escapeHtml(created)}</span>` : ""}
      <span>状态：草稿</span>
    </div>
    ${noteText ? `<div class="cit"><b>说明</b><div>${escapeHtml(noteText)}</div></div>` : ""}
    ${parts.join("")}
  `;
}

async function pollOnce() {
  if (!currentPlanId) return;
  if ($("statusErr")) $("statusErr").textContent = "";
  try {
    const j = await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}`);
    if ($("plan_id")) $("plan_id").textContent = j.plan_id || currentPlanId;
    if ($("status")) $("status").textContent = j.status || "";
    if ($("output_path")) $("output_path").textContent = j.output_path || "";
    if ($("plan_text")) $("plan_text").textContent = j.plan_text || "";
    if (j.error && $("statusErr")) $("statusErr").textContent = j.error;
    if (j.output_path) {
      const base = getApiBase();
      const dl = `${base}/cdc/plan/${encodeURIComponent(currentPlanId)}/download`;
      if ($("download")) {
        $("download").href = dl;
        $("download").style.display = "inline";
        const lower = String(j.output_path || "").toLowerCase();
        if (lower.endsWith(".pdf")) $("download").textContent = "下载 PDF（.pdf）";
        else $("download").textContent = "下载 Word（.docx）";
      }
      if ($("btnFetchPlanJson")) $("btnFetchPlanJson").disabled = false;
    }
    if (j.status === "waiting_manual_fix") {
      await refreshManualFixUI();
    } else {
      showManualFix(false);
    }
    if (j.status === "waiting_review") {
      await refreshReviewUI();
    } else {
      showReview(false);
    }
    if (j.status === "done" || j.status === "error") {
      stopPolling();
    }
  } catch (e) {
    if ($("statusErr")) $("statusErr").textContent = String(e && e.message ? e.message : e);
  }
}

async function run() {
  clearRunUI();
  $("apiBase").value = getApiBase();
  setBusy(true);
  if ($("runErr")) $("runErr").textContent = "";
  try {
    const payload = buildPayload();
    const j = await apiFetch("/cdc/plan/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    currentPlanId = j.plan_id;
    if (currentPage === "dashboard") {
      window.location.href = `./preview.html?plan_id=${encodeURIComponent(currentPlanId)}`;
      return;
    }
    if ($("plan_id")) $("plan_id").textContent = currentPlanId;
    await pollOnce();
    timer = setInterval(pollOnce, 2000);
  } catch (e) {
    stopPolling();
    if ($("runErr")) $("runErr").textContent = String(e && e.message ? e.message : e);
  }
}

async function fetchPlanJson() {
  if (!currentPlanId) return;
  if ($("statusErr")) $("statusErr").textContent = "";
  try {
    const j = await apiFetch(`/cdc/plan/${encodeURIComponent(currentPlanId)}/plan.json`);
    if ($("plan_json")) $("plan_json").textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    if ($("statusErr")) $("statusErr").textContent = String(e && e.message ? e.message : e);
  }
}

async function refreshRuns() {
  if (!$("runsBody")) return;
  $("runsErr").textContent = "";
  const tbody = $("runsBody");
  tbody.innerHTML = "";
  try {
    const j = await apiFetch("/cdc/runs");
    const runs = (j && j.runs) || [];
    for (const r of runs) {
      const tr = document.createElement("tr");
      const tdId = document.createElement("td");
      tdId.textContent = r.plan_id || "";
      const tdSt = document.createElement("td");
      tdSt.textContent = r.status || "";
      const tdDl = document.createElement("td");
      if (r.output_path) {
        const a = document.createElement("a");
        a.href = `${getApiBase()}/cdc/plan/${encodeURIComponent(r.plan_id)}/download`;
        a.target = "_blank";
        a.textContent = "下载";
        tdDl.appendChild(a);
      } else {
        tdDl.textContent = "-";
      }
      const tdErr = document.createElement("td");
      tdErr.textContent = r.error || "";
      tr.appendChild(tdId);
      tr.appendChild(tdSt);
      tr.appendChild(tdDl);
      tr.appendChild(tdErr);
      tbody.appendChild(tr);
    }
  } catch (e) {
    $("runsErr").textContent = String(e && e.message ? e.message : e);
  }
}

async function openPid() {
  const pid = ($("pidInput")?.value || "").trim();
  if (!pid) return;
  currentPlanId = pid;
  if ($("onePlanMeta")) $("onePlanMeta").textContent = `plan_id：${pid}`;
  try {
    const j = await apiFetch(`/cdc/plan/${encodeURIComponent(pid)}`);
    if ($("onePlanText")) $("onePlanText").textContent = j.plan_text || "";
  } catch (e) {
    if ($("onePlanText")) $("onePlanText").textContent = String(e && e.message ? e.message : e);
  }
}

function init() {
  const defaultBase = `http://${window.location.hostname}:8000`;
  if ($("apiBase")) $("apiBase").value = localStorage.getItem("cdc_api_base") || defaultBase;

  if ($("btnRun")) $("btnRun").addEventListener("click", (e) => { e.preventDefault(); run(); });
  if ($("btnStop")) $("btnStop").addEventListener("click", (e) => { e.preventDefault(); stopPolling(); });
  if ($("btnReload")) $("btnReload").addEventListener("click", (e) => { e.preventDefault(); pollOnce(); });
  if ($("btnFetchPlanJson")) $("btnFetchPlanJson").addEventListener("click", (e) => { e.preventDefault(); fetchPlanJson(); });

  if ($("manualFixCard")) {
    showManualFix(false);
    $("btnMF_S").addEventListener("click", (e) => { e.preventDefault(); submitManualFix("s"); });
    $("btnMF_X").addEventListener("click", (e) => { e.preventDefault(); submitManualFix("x"); });
    $("btnMF_Q").addEventListener("click", (e) => { e.preventDefault(); submitManualFix("q"); });
    $("btnMF_E").addEventListener("click", (e) => {
      e.preventDefault();
      let obj;
      try {
        obj = JSON.parse(($("manualFixText").value || "{}"));
      } catch {
        $("manualFixErr").textContent = "预案 JSON 不是合法 JSON。";
        return;
      }
      submitManualFix("e", obj);
    });
  }

  if ($("btnRefreshRuns")) $("btnRefreshRuns").addEventListener("click", (e) => { e.preventDefault(); refreshRuns(); });
  if ($("btnClearView")) $("btnClearView").addEventListener("click", (e) => { e.preventDefault(); if ($("onePlanText")) $("onePlanText").textContent = ""; if ($("onePlanMeta")) $("onePlanMeta").textContent = ""; });
  if ($("btnOpenPid")) $("btnOpenPid").addEventListener("click", (e) => { e.preventDefault(); openPid(); });
  if ($("runsBody")) refreshRuns();

  if (currentPage === "preview") {
    const url = new URL(window.location.href);
    const pid = url.searchParams.get("plan_id");
    if (pid) {
      currentPlanId = pid;
      if ($("previewPlanId")) $("previewPlanId").textContent = pid;
      setBusy(true);
      pollOnce();
      timer = setInterval(pollOnce, 2000);
    }
    if ($("btnToggleRaw")) $("btnToggleRaw").addEventListener("click", (e) => {
      e.preventDefault();
      const el = $("draftRaw");
      if (!el) return;
      el.style.display = el.style.display === "none" ? "block" : "none";
    });
    if ($("btnApprove")) $("btnApprove").addEventListener("click", (e) => { e.preventDefault(); submitReview("approve"); });
    if ($("btnCancel")) $("btnCancel").addEventListener("click", (e) => { e.preventDefault(); submitReview("cancel"); });
    if ($("btnEditExport")) $("btnEditExport").addEventListener("click", (e) => {
      e.preventDefault();
      let obj;
      try {
        obj = JSON.parse(($("draftJson")?.value || "{}"));
      } catch {
        if ($("reviewErr")) $("reviewErr").textContent = "草稿 JSON 不是合法 JSON。";
        return;
      }
      submitReview("edit", obj);
    });
  }
}

init();
