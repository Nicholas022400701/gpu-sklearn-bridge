/* ═══════════════════════════════════════════════════════════════════════════
   cuML WebUI – Main Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
const STATE = {
  sessionId: null,          // data session
  dataLoaded: false,
  splitDone: false,
  selectedModel: null,
  modelParams: {},
  trainSession: null,       // active training session id
  trainStatus: "idle",
  charts: {},               // Chart.js instances
  trainMode: "unlimited",
  activeModule: "data",
};

// ── Socket.IO ─────────────────────────────────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });

socket.on("connect",     () => updateConnectionStatus(true));
socket.on("disconnect",  () => updateConnectionStatus(false));

socket.on("training_start", (data) => {
  STATE.trainStatus = "running";
  updateTrainingUI();
  notify("Training started", "info");
});

socket.on("training_step", (data) => {
  appendTrainingStep(data);
});

socket.on("training_done", (data) => {
  STATE.trainStatus = data.status;
  updateTrainingUI();
  notify(`Training ${data.status}. Best: ${JSON.stringify(data.best_metrics)}`, "success");
  refreshWeightsList();
});

socket.on("training_error", (data) => {
  STATE.trainStatus = "error";
  updateTrainingUI();
  notify("Training error: " + data.error, "error");
});

socket.on("log_entry", (data) => {
  if (data.model_id === document.getElementById("log-model-select")?.value) {
    appendLogEntry(data.entry);
  }
});

// ── Navigation ────────────────────────────────────────────────────────────────
function showModule(name) {
  document.querySelectorAll(".module").forEach(m => m.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  const mod = document.getElementById("module-" + name);
  if (mod) mod.classList.add("active");
  const nav = document.querySelector(`.nav-item[data-module="${name}"]`);
  if (nav) nav.classList.add("active");
  STATE.activeModule = name;

  // Lazy-load
  if (name === "weights")   refreshWeightsList();
  if (name === "logs")      refreshLogsList();
  if (name === "training")  renderSessionList();
  if (name === "predict")   refreshPredictModelList();
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => showModule(item.dataset.module));
});

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const dot  = document.getElementById("bridge-dot");
    const text = document.getElementById("bridge-text");
    if (d.bridge_connected) {
      dot.className  = "status-dot ok";
      text.textContent = "GPU Bridge connected";
    } else {
      dot.className  = "status-dot warn";
      text.textContent = "Standalone (sklearn)";
    }
  } catch {
    document.getElementById("bridge-dot").className = "status-dot error";
    document.getElementById("bridge-text").textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 10000);

function updateConnectionStatus(ok) {
  const el = document.getElementById("ws-status");
  if (el) el.textContent = ok ? "WS ✓" : "WS ✗";
}

// ── Notifications ─────────────────────────────────────────────────────────────
function notify(msg, type = "info", duration = 4000) {
  const container = document.getElementById("notifications");
  const el = document.createElement("div");
  el.className = `notif ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: DATA
// ─────────────────────────────────────────────────────────────────────────────

// Upload zone
const uploadZone = document.getElementById("upload-zone");
const fileInput  = document.getElementById("file-input");

if (uploadZone) {
  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("dragover"); });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", e => {
    e.preventDefault(); uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
}

if (fileInput) fileInput.addEventListener("change", e => { if (e.target.files.length) uploadFile(e.target.files[0]); });

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  if (STATE.sessionId) fd.append("session_id", STATE.sessionId);

  uploadZone.innerHTML = `<div class="spinner"></div><div style="margin-top:10px;color:var(--text-dim)">Uploading…</div>`;

  try {
    const r = await fetch("/api/data/upload", { method: "POST", body: fd });
    const d = await r.json();
    if (d.error) { notify(d.error, "error"); resetUploadZone(); return; }

    STATE.sessionId = d.session_id;
    STATE.dataLoaded = true;
    renderDataSummary(d);
    renderDataPreview(d);
    renderColumnSelector(d);
    notify(`Loaded ${d.n_rows} rows × ${d.n_cols} cols`, "success");
    resetUploadZone();
  } catch (e) {
    notify("Upload failed: " + e.message, "error");
    resetUploadZone();
  }
}

function resetUploadZone() {
  uploadZone.innerHTML = `
    <div class="uz-icon">📂</div>
    <div class="uz-text">Drop a file here or click to browse</div>
    <div class="uz-hint">CSV · TSV · Excel · JSON · Parquet</div>`;
}

function renderDataSummary(d) {
  const el = document.getElementById("data-summary");
  if (!el) return;
  el.innerHTML = `
    <div class="metric-grid">
      <div class="metric-card"><div class="label">Rows</div><div class="value neutral">${d.n_rows.toLocaleString()}</div></div>
      <div class="metric-card"><div class="label">Columns</div><div class="value neutral">${d.n_cols}</div></div>
      <div class="metric-card"><div class="label">Missing Cells</div>
        <div class="value ${d.columns.reduce((a,c)=>a+c.n_missing,0)>0?'bad':''}">
          ${d.columns.reduce((a,c)=>a+c.n_missing,0).toLocaleString()}</div></div>
      <div class="metric-card"><div class="label">Session</div>
        <div class="value" style="font-size:12px;word-break:break-all">${STATE.sessionId?.slice(0,12)||'–'}</div></div>
    </div>`;
}

function renderDataPreview(d) {
  const el = document.getElementById("data-preview-table");
  if (!el) return;
  const headers = d.col_names.map(c => `<th>${c}</th>`).join("");
  const rows = d.preview.map(row => `<tr>${row.map(v=>`<td>${v}</td>`).join("")}</tr>`).join("");
  el.innerHTML = `<table class="data-table"><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderColumnSelector(d) {
  const featureDiv = document.getElementById("feature-col-list");
  const targetSel  = document.getElementById("target-col-select");
  if (!featureDiv || !targetSel) return;

  featureDiv.innerHTML = d.col_names.map(c => `
    <div class="checkbox-row" style="margin-bottom:4px">
      <input type="checkbox" id="fc_${c}" name="feature_cols" value="${c}" checked>
      <label for="fc_${c}" style="font-size:12px">${c}</label>
      <span class="tag" style="margin-left:auto;font-size:10px">${d.columns.find(x=>x.name===c)?.dtype||''}</span>
    </div>`).join("");

  targetSel.innerHTML = `<option value="">— No target (unsupervised) —</option>` +
    d.col_names.map(c => `<option value="${c}">${c}</option>`).join("");

  // Guess target: last column
  if (d.col_names.length) targetSel.value = d.col_names[d.col_names.length - 1];
}

// Column info table
function renderColumnInfo(d) {
  const el = document.getElementById("col-info-table");
  if (!el) return;
  const rows = d.columns.map(c => `
    <tr>
      <td><strong>${c.name}</strong></td>
      <td><span class="tag">${c.dtype}</span></td>
      <td>${c.n_missing} <span style="color:var(--text-dim)">(${c.pct_missing}%)</span></td>
      <td>${c.n_unique}</td>
      <td style="font-size:11px;color:var(--text-dim)">${c.sample.slice(0,3).join(", ")}</td>
    </tr>`).join("");
  el.innerHTML = `<table class="data-table">
    <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Sample</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ── Cleaning ──────────────────────────────────────────────────────────────────
async function applyCleaning() {
  if (!STATE.sessionId) { notify("Upload data first", "error"); return; }
  const ops = [];

  if (document.getElementById("clean-drop-dup")?.checked)
    ops.push({ type: "drop_duplicates" });

  const missingStrategy = document.getElementById("clean-missing-strategy")?.value;
  if (missingStrategy && missingStrategy !== "none")
    ops.push({ type: "fill_missing", strategy: missingStrategy,
               value: document.getElementById("clean-fill-value")?.value || 0 });

  const outlierMethod = document.getElementById("clean-outlier-method")?.value;
  if (outlierMethod && outlierMethod !== "none")
    ops.push({ type: "clip_outliers", method: outlierMethod,
               threshold: parseFloat(document.getElementById("clean-outlier-thresh")?.value || 1.5) });

  try {
    const r = await fetch("/api/data/clean", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: STATE.sessionId, ops }),
    });
    const d = await r.json();
    if (d.error) { notify(d.error, "error"); return; }
    renderDataSummary(d);
    renderDataPreview(d);
    renderColumnInfo(d);
    notify("Cleaning applied: " + d.applied.join("; "), "success");
  } catch (e) { notify("Error: " + e.message, "error"); }
}

async function resetCleaning() {
  if (!STATE.sessionId) return;
  await fetch("/api/data/clean", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: STATE.sessionId, ops: [{ type: "reset" }] }),
  });
  const d = await (await fetch(`/api/data/describe?session_id=${STATE.sessionId}`)).json();
  renderDataSummary(d);
  renderDataPreview(d);
  renderColumnInfo(d);
  notify("Data reset to original", "info");
}

// ── Preprocessing / Split ─────────────────────────────────────────────────────
async function applyPreprocess() {
  if (!STATE.sessionId) { notify("Upload data first", "error"); return; }

  const featureCols = [...document.querySelectorAll("input[name='feature_cols']:checked")]
    .map(c => c.value);
  const targetCol = document.getElementById("target-col-select")?.value || null;
  const testSize  = parseFloat(document.getElementById("test-size")?.value || 0.2);
  const seed      = parseInt(document.getElementById("random-seed")?.value || 42);
  const scaler    = document.getElementById("preprocess-scaler")?.value || null;
  const encodeTarget = document.getElementById("encode-target")?.checked || false;

  if (!featureCols.length) { notify("Select at least one feature column", "error"); return; }

  try {
    const r = await fetch("/api/data/split", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: STATE.sessionId,
        feature_cols: featureCols,
        target_col: targetCol || null,
        test_size: testSize,
        random_state: seed,
        scaler: scaler || null,
        encode_target: encodeTarget,
      }),
    });
    const d = await r.json();
    if (d.error) { notify(d.error, "error"); return; }

    STATE.splitDone = true;
    document.getElementById("split-result").innerHTML = `
      <div class="metric-grid">
        <div class="metric-card"><div class="label">Train</div><div class="value">${d.n_train.toLocaleString()}</div></div>
        <div class="metric-card"><div class="label">Test</div><div class="value">${d.n_test.toLocaleString()}</div></div>
        <div class="metric-card"><div class="label">Features</div><div class="value">${d.n_features}</div></div>
        <div class="metric-card"><div class="label">Target</div><div class="value neutral" style="font-size:14px">${d.target_col||'–'}</div></div>
      </div>
      ${d.label_map ? `<div style="margin-top:10px;font-size:12px;color:var(--text-dim)">Label map: ${JSON.stringify(d.label_map)}</div>` : ''}`;
    notify("Data split ready. Train=" + d.n_train + " Test=" + d.n_test, "success");
  } catch (e) { notify("Error: " + e.message, "error"); }
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: MODEL SELECTION
// ─────────────────────────────────────────────────────────────────────────────
let ALL_MODELS = [];

async function loadModelList() {
  const r = await fetch("/api/models/list");
  ALL_MODELS = await r.json();
  renderModelCards(ALL_MODELS);
}
loadModelList();

function renderModelCards(models, filterTask = "all") {
  const grid = document.getElementById("model-card-grid");
  if (!grid) return;
  const filtered = filterTask === "all" ? models : models.filter(m => m.task === filterTask);
  grid.innerHTML = filtered.map(m => `
    <div class="model-card ${STATE.selectedModel===m.name?'selected':''}"
         data-model="${m.name}" onclick="selectModel('${m.name}')">
      <div class="mc-name">${m.name}</div>
      <div class="mc-task">
        <span class="tag ${taskColor(m.task)}">${m.task}</span>
        <span style="margin-left:4px;font-size:11px;color:#555">${m.module}</span>
      </div>
      <div class="mc-desc">${m.desc}</div>
    </div>`).join("");
}

function taskColor(task) {
  return { classification:"green", regression:"blue", clustering:"yellow", decomposition:"", preprocessing:"", unsupervised:"" }[task] || "";
}

document.querySelectorAll(".model-filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".model-filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderModelCards(ALL_MODELS, btn.dataset.task || "all");
  });
});

async function selectModel(name) {
  STATE.selectedModel = name;
  STATE.modelParams = {};
  document.querySelectorAll(".model-card").forEach(c => c.classList.toggle("selected", c.dataset.model === name));
  await renderParamPanel(name);
  document.getElementById("selected-model-display").textContent = name;
  document.getElementById("selected-model-display2").textContent = name;
}

async function renderParamPanel(modelName) {
  const panel = document.getElementById("param-panel");
  if (!panel) return;
  const r = await fetch("/api/models/configs");
  const cfgs = await r.json();
  const cfg = cfgs[modelName];
  if (!cfg) { panel.innerHTML = "<p>No config found.</p>"; return; }

  const task = cfg.task;
  const taskMetricsEl = document.getElementById("task-metrics-info");
  if (taskMetricsEl) {
    const metrics = { classification:["accuracy","f1"], regression:["r2","rmse"], clustering:["silhouette","inertia"] };
    taskMetricsEl.textContent = "Monitored metrics: " + (metrics[task]||["score"]).join(", ");
  }

  const params = cfg.params || {};
  if (!Object.keys(params).length) {
    panel.innerHTML = `<p style="color:var(--text-dim);font-size:13px">No configurable parameters for ${modelName}.</p>`;
    return;
  }

  const fields = Object.entries(params).map(([k, p]) => {
    const id = `param_${k}`;
    let input = "";
    const val = p.default !== undefined && p.default !== null ? p.default : "";

    if (p.type === "bool") {
      input = `<div class="checkbox-row">
        <input type="checkbox" id="${id}" name="${k}" ${val ? "checked" : ""} onchange="updateParam('${k}', this.checked)">
        <label for="${id}" style="font-size:13px;text-transform:none">${p.desc}</label>
      </div>`;
    } else if (p.type === "select") {
      input = `<select id="${id}" onchange="updateParam('${k}', this.value)">
        ${(p.options||[]).map(o => `<option value="${o}" ${o===val?"selected":""}>${o}</option>`).join("")}
      </select>`;
    } else if (p.type === "tuple_float") {
      const arr = Array.isArray(val) ? val : [0,1];
      input = `<div style="display:flex;gap:8px">
        <input type="number" step="0.01" value="${arr[0]}" placeholder="min" style="width:50%"
               onchange="updateParam('${k}', [parseFloat(this.value), parseFloat(document.getElementById('${id}_max').value)])">
        <input id="${id}_max" type="number" step="0.01" value="${arr[1]}" placeholder="max" style="width:50%"
               onchange="updateParam('${k}', [parseFloat(document.getElementById('${id}').value), parseFloat(this.value)])">
      </div>`;
    } else {
      const min = p.min !== undefined ? `min="${p.min}"` : "";
      const max = p.max !== undefined ? `max="${p.max}"` : "";
      const step = p.type === "float" ? `step="any"` : "";
      const nullable = p.nullable ? `placeholder="null"` : `placeholder="${val}"`;
      input = `<input id="${id}" type="number" ${min} ${max} ${step} value="${val !== null && val !== undefined ? val : ''}"
                      ${nullable} oninput="updateParam('${k}', this.value)">`;
    }

    // Pre-populate STATE.modelParams with default
    if (p.type !== "bool" && p.type !== "tuple_float") {
      STATE.modelParams[k] = val;
    } else if (p.type === "bool") {
      STATE.modelParams[k] = val;
    } else {
      STATE.modelParams[k] = val;
    }

    return `<div class="form-group">
      <label for="${id}">${k}</label>
      ${input}
      ${p.type !== "bool" ? `<div class="hint">${p.desc}</div>` : ''}
    </div>`;
  });

  panel.innerHTML = `<div class="param-grid">${fields.join("")}</div>`;
}

function updateParam(key, value) {
  STATE.modelParams[key] = value;
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: TRAINING
// ─────────────────────────────────────────────────────────────────────────────

// Mode selection
document.querySelectorAll(".mode-card").forEach(card => {
  card.addEventListener("click", () => {
    document.querySelectorAll(".mode-card").forEach(c => c.classList.remove("selected"));
    card.classList.add("selected");
    STATE.trainMode = card.dataset.mode;
    updateModeOptions();
  });
});

function updateModeOptions() {
  document.getElementById("timed-options").style.display   = STATE.trainMode === "timed"  ? "block" : "none";
  document.getElementById("target-options").style.display  = STATE.trainMode === "target" ? "block" : "none";
}

async function startTraining() {
  if (!STATE.selectedModel) { notify("Select a model first", "error"); return; }
  if (!STATE.splitDone)      { notify("Prepare data split first", "error"); return; }

  const modelName = document.getElementById("train-model-name")?.value || STATE.selectedModel;
  const nSteps    = parseInt(document.getElementById("train-n-steps")?.value || 10);

  const payload = {
    session_id:    "train_" + Date.now(),
    data_session:  STATE.sessionId,
    model:         STATE.selectedModel,
    params:        STATE.modelParams,
    mode:          STATE.trainMode,
    n_steps:       nSteps,
    model_name:    modelName,
  };

  if (STATE.trainMode === "timed") {
    payload.time_limit = parseFloat(document.getElementById("time-limit")?.value || 60);
  }
  if (STATE.trainMode === "target") {
    payload.target_metric = document.getElementById("target-metric")?.value || "val_accuracy";
    payload.target_value  = parseFloat(document.getElementById("target-value")?.value || 0.95);
  }

  try {
    const r = await fetch("/api/train/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (d.error) { notify(d.error, "error"); return; }

    STATE.trainSession = d.session_id;
    STATE.trainStatus  = "running";

    socket.emit("join_session", { session_id: d.session_id });
    updateTrainingUI();
    resetCharts();
    notify(`Training started: ${STATE.selectedModel}`, "info");
    document.getElementById("train-current-session").textContent = d.session_id;
  } catch (e) { notify("Error: " + e.message, "error"); }
}

async function stopTraining() {
  if (!STATE.trainSession) return;
  await fetch("/api/train/stop", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: STATE.trainSession }),
  });
  notify("Stop signal sent", "warning");
}

function updateTrainingUI() {
  const btn  = document.getElementById("btn-start-train");
  const btnS = document.getElementById("btn-stop-train");
  const statusEl = document.getElementById("train-status-label");

  const statusColors = { running: "green", stopped: "yellow", completed: "green", error: "red", idle: "" };
  if (statusEl) {
    statusEl.textContent = STATE.trainStatus.toUpperCase();
    statusEl.className = "tag " + (statusColors[STATE.trainStatus] || "");
  }
  if (btn)  btn.disabled  = STATE.trainStatus === "running";
  if (btnS) btnS.disabled = STATE.trainStatus !== "running";
}

// Training step handler
function appendTrainingStep(data) {
  const m = data.metrics || {};
  const step = data.step;
  const elapsed = data.elapsed;

  // Update elapsed timer
  const el = document.getElementById("train-elapsed");
  if (el) el.textContent = elapsed + "s";

  // Update best metrics display
  const best = data.best_metrics || {};
  const bmEl = document.getElementById("best-metrics-display");
  if (bmEl) {
    const keys = Object.keys(best).filter(k => !k.startsWith("_"));
    bmEl.innerHTML = keys.map(k => `
      <div class="metric-card">
        <div class="label">${k}</div>
        <div class="value ${best[k]<0?'bad':''}">${typeof best[k]==='number'?best[k].toFixed(4):best[k]}</div>
      </div>`).join("");
  }

  // Update current metrics
  const cmEl = document.getElementById("current-metrics-display");
  if (cmEl) {
    const keys = Object.keys(m).filter(k => !k.startsWith("_"));
    cmEl.innerHTML = keys.map(k => `
      <div class="metric-card">
        <div class="label">${k}</div>
        <div class="value neutral ${m[k]<0?'bad':''}">${typeof m[k]==='number'?m[k].toFixed(4):m[k]}</div>
      </div>`).join("");
  }

  // Feed charts
  updateCharts(step, m);

  // Progress bar
  const progEl = document.getElementById("train-progress-fill");
  if (progEl && data.frac) progEl.style.width = (data.frac * 100) + "%";
}

// ── Charts (Chart.js) ─────────────────────────────────────────────────────────
function initCharts() {
  if (typeof Chart === "undefined") return;
  const commonOpts = {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 100 },
    plugins: { legend: { labels: { color: "#888", font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#555" }, grid: { color: "#222" } },
      y: { ticks: { color: "#555" }, grid: { color: "#222" }, beginAtZero: false },
    },
  };

  const makeChart = (id, label1, label2, color1, color2) => {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    return new Chart(canvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: label1, data: [], borderColor: color1, backgroundColor: color1 + "20", tension: 0.3, pointRadius: 2 },
          { label: label2, data: [], borderColor: color2, backgroundColor: color2 + "20", tension: 0.3, pointRadius: 2 },
        ],
      },
      options: { ...commonOpts },
    });
  };

  STATE.charts.accuracy = makeChart("chart-accuracy", "Train Accuracy", "Val Accuracy", "#76b900", "#00b2ff");
  STATE.charts.loss     = makeChart("chart-loss",     "Train RMSE",     "Val RMSE",     "#ff6b35", "#ffcc00");
  STATE.charts.f1       = makeChart("chart-f1",       "Train F1",       "Val F1",       "#76b900", "#00b2ff");
  STATE.charts.r2       = makeChart("chart-r2",       "Train R²",       "Val R²",       "#76b900", "#00b2ff");
}

function resetCharts() {
  Object.values(STATE.charts).forEach(c => {
    if (!c) return;
    c.data.labels = [];
    c.data.datasets.forEach(ds => ds.data = []);
    c.update();
  });
}

function updateCharts(step, metrics) {
  const push = (chart, x, y0, y1) => {
    if (!chart) return;
    chart.data.labels.push(x);
    if (y0 !== undefined) chart.data.datasets[0].data.push(y0);
    if (y1 !== undefined) chart.data.datasets[1].data.push(y1);
    chart.update("none");
  };

  push(STATE.charts.accuracy, step, metrics.train_accuracy, metrics.val_accuracy);
  push(STATE.charts.loss,     step, metrics.train_rmse,     metrics.val_rmse);
  push(STATE.charts.f1,       step, metrics.train_f1,       metrics.val_f1);
  push(STATE.charts.r2,       step, metrics.train_r2,       metrics.val_r2);
}

document.addEventListener("DOMContentLoaded", initCharts);

// ── Session list ──────────────────────────────────────────────────────────────
async function renderSessionList() {
  const el = document.getElementById("session-list");
  if (!el) return;
  const r = await fetch("/api/train/sessions");
  const sessions = await r.json();
  if (!sessions.length) {
    el.innerHTML = `<div class="empty-state"><div class="es-icon">🏋</div><div class="es-text">No training sessions yet.</div></div>`;
    return;
  }
  el.innerHTML = `<table class="data-table">
    <thead><tr><th>Session</th><th>Model</th><th>Status</th><th>Mode</th><th>Best Metrics</th></tr></thead>
    <tbody>${sessions.map(s => `<tr>
      <td style="font-size:11px;font-family:monospace">${s.session_id.slice(0,16)}…</td>
      <td>${s.model}</td>
      <td><span class="tag ${s.status==='completed'?'green':s.status==='running'?'blue':s.status==='error'?'red':'yellow'}">${s.status}</span></td>
      <td>${s.mode}</td>
      <td style="font-size:11px">${Object.entries(s.best_metrics||{}).filter(([k])=>!k.startsWith('_')).map(([k,v])=>`${k}:${typeof v==='number'?v.toFixed(3):v}`).join(', ')||'–'}</td>
    </tr>`).join("")}</tbody></table>`;
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: WEIGHTS
// ─────────────────────────────────────────────────────────────────────────────
async function refreshWeightsList() {
  const el = document.getElementById("weights-list");
  if (!el) return;
  const r = await fetch("/api/weights/list");
  const d = await r.json();
  const models = d.models || [];

  // Also refresh predict model dropdown
  const sel = document.getElementById("predict-model-select");
  if (sel) {
    sel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join("");
  }

  const evalSel = document.getElementById("eval-model-select");
  if (evalSel) evalSel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join("");

  // Also populate stacking base model checklist
  const stackList = document.getElementById("stack-base-list");
  if (stackList) {
    stackList.innerHTML = models.map(m => `
      <div class="checkbox-row" style="margin-bottom:6px">
        <input type="checkbox" id="sm_${m}" name="stack_model" value="${m}">
        <label for="sm_${m}">${m}</label>
      </div>`).join("");
  }

  if (!models.length) {
    el.innerHTML = `<div class="empty-state"><div class="es-icon">💾</div><div class="es-text">No saved models yet. Train a model and it will auto-save here.</div></div>`;
    return;
  }

  el.innerHTML = `<table class="data-table">
    <thead><tr><th>Model Name</th><th>Actions</th></tr></thead>
    <tbody>${models.map(m => `<tr>
      <td><span style="font-weight:600">${m}</span></td>
      <td>
        <div class="btn-group">
          <button class="btn btn-sm btn-secondary" onclick="loadModelForUse('${m}')">⬆ Load</button>
          <button class="btn btn-sm btn-danger"    onclick="deleteModel('${m}')">✕ Delete</button>
        </div>
      </td>
    </tr>`).join("")}</tbody></table>`;
}

async function loadModelForUse(name) {
  const r = await fetch("/api/weights/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const d = await r.json();
  if (d.error) { notify(d.error, "error"); return; }
  notify(`Model '${name}' loaded (${d.class})`, "success");
}

async function deleteModel(name) {
  if (!confirm(`Delete model '${name}'?`)) return;
  const r = await fetch("/api/weights/delete", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const d = await r.json();
  if (d.ok) { notify(`'${name}' deleted`, "info"); refreshWeightsList(); }
  else notify("Delete failed", "error");
}

function refreshPredictModelList() {
  refreshWeightsList(); // Same data, already handled
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: EVALUATION
// ─────────────────────────────────────────────────────────────────────────────
async function runEvaluation() {
  const modelId = document.getElementById("eval-model-select")?.value;
  if (!modelId) { notify("Select a model", "error"); return; }
  if (!STATE.sessionId) { notify("Load data first", "error"); return; }

  const r = await fetch("/api/evaluate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, data_session: STATE.sessionId }),
  });
  const d = await r.json();
  if (d.error) { notify(d.error, "error"); return; }

  const el = document.getElementById("eval-metrics-display");
  if (el) {
    const m = d.metrics || {};
    el.innerHTML = `<div class="metric-grid">
      ${Object.entries(m).filter(([k])=>!k.startsWith('_')).map(([k,v])=>`
        <div class="metric-card">
          <div class="label">${k}</div>
          <div class="value ${v<0?'bad':''}">${typeof v==='number'?v.toFixed(4):v}</div>
        </div>`).join("")}
    </div>`;
  }
  notify("Evaluation complete", "success");
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: PREDICTION
// ─────────────────────────────────────────────────────────────────────────────
async function runPrediction() {
  const modelId = document.getElementById("predict-model-select")?.value;
  if (!modelId) { notify("Select a model", "error"); return; }
  if (!STATE.sessionId) { notify("Load data first", "error"); return; }

  const r = await fetch("/api/predict", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, data_session: STATE.sessionId }),
  });
  const d = await r.json();
  if (d.error) { notify(d.error, "error"); return; }

  const el = document.getElementById("pred-results");
  if (el) {
    const preds = d.predictions || [];
    el.innerHTML = `
      <div style="margin-bottom:8px;color:var(--text-dim);font-size:12px">${preds.length} predictions</div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>#</th><th>Prediction</th></tr></thead>
          <tbody>${preds.slice(0, 200).map((p,i) => `<tr><td>${i+1}</td><td><strong>${p}</strong></td></tr>`).join("")}</tbody>
        </table>
      </div>`;
  }
  notify(`${d.n} predictions generated`, "success");
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: STACKING
// ─────────────────────────────────────────────────────────────────────────────
async function createStack() {
  const baseModels = [...document.querySelectorAll("input[name='stack_model']:checked")].map(c => c.value);
  if (baseModels.length < 2) { notify("Select at least 2 base models", "error"); return; }
  const metaModel = document.getElementById("stack-meta-model")?.value || "LogisticRegression";
  const stackName = document.getElementById("stack-name")?.value || ("stack_" + Date.now());
  if (!STATE.sessionId) { notify("Load data first", "error"); return; }

  const r = await fetch("/api/stacking/create", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_models: baseModels,
      meta_model: metaModel,
      name: stackName,
      data_session: STATE.sessionId,
    }),
  });
  const d = await r.json();
  if (d.error) { notify(d.error, "error"); return; }

  notify(`Stacking model '${stackName}' created!`, "success");
  const el = document.getElementById("stack-result");
  if (el) {
    const m = d.metrics || {};
    el.innerHTML = `<div class="card"><div class="card-title">Stacking Results</div>
      <div class="metric-grid">
        ${Object.entries(m).filter(([k])=>!k.startsWith('_')).map(([k,v])=>`
          <div class="metric-card"><div class="label">${k}</div>
          <div class="value">${typeof v==='number'?v.toFixed(4):v}</div></div>`).join("")}
      </div></div>`;
  }
  refreshWeightsList();
}

// ── ─────────────────────────────────────────────────────────────────────────
// MODULE: LOGS
// ─────────────────────────────────────────────────────────────────────────────
async function refreshLogsList() {
  const sel = document.getElementById("log-model-select");
  if (!sel) return;
  const r = await fetch("/api/logs/list");
  const d = await r.json();
  const logs = d.logs || [];
  sel.innerHTML = logs.map(l => `<option value="${l}">${l}</option>`).join("");
  if (logs.length) loadLog(logs[0]);
}

async function loadLog(modelId) {
  if (!modelId) return;
  const r = await fetch(`/api/logs/${modelId}`);
  const d = await r.json();
  const el = document.getElementById("log-viewer");
  if (!el) return;
  el.innerHTML = "";
  (d.entries || []).forEach(e => appendLogEntry(e, el));
  el.scrollTop = el.scrollHeight;
}

function appendLogEntry(entry, container) {
  const el = container || document.getElementById("log-viewer");
  if (!el) return;
  const line = document.createElement("div");
  line.className = `log-line ${entry.level}`;
  line.innerHTML = `<span class="log-ts">${entry.ts}</span><span class="log-level ${entry.level}">[${entry.level}]</span>${escapeHtml(entry.msg)}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

async function clearLog() {
  const modelId = document.getElementById("log-model-select")?.value;
  if (!modelId) return;
  await fetch(`/api/logs/${modelId}/clear`, { method: "DELETE" });
  document.getElementById("log-viewer").innerHTML = "";
  notify("Log cleared", "info");
}

function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Tab switching within modules ──────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    const parent = btn.closest(".card") || document.body;
    parent.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    parent.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const pane = parent.querySelector(`.tab-pane[data-tab="${target}"]`);
    if (pane) pane.classList.add("active");
  });
});

// ── Initial render ────────────────────────────────────────────────────────────
showModule("data");
updateModeOptions();
