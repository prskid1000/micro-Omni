/* ── micro-Omni Dashboard — app.js ───────────────────────── */
"use strict";

// ── State ───────────────────────────────────────────────────
const state = {
  live: true,
  pollMs: 5000,
  pollTimer: null,
  smoothing: 0.6,
  logScale: false,
  dualAxis: false,
  allRows: [],          // flattened metric rows across all files
  fileList: [],
  chart: null,
  lastTimestamp: null,
  pipelineData: null,
  gpuData: null,
  summaryData: null,
  checkpointData: null,
};

// ── API client ──────────────────────────────────────────────
const api = {
  get: (url) => fetch(url, { cache: "no-store" }).then(r => r.json()).catch(() => ({ ok: false })),
  post: (url, body) => fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(r => r.json()).catch(() => ({ ok: false })),
};

// ── DOM refs ────────────────────────────────────────────────
const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const dom = {
  gpuBadge:      $("#gpuBadge"),
  liveToggle:    $("#liveToggle"),
  pipeline:      $("#pipeline"),
  fileSelect:    $("#fileSelect"),
  runSelect:     $("#runSelect"),
  metricSelect:  $("#metricSelect"),
  xAxisSelect:   $("#xAxisSelect"),
  xAxisToggle:   $("#xAxisToggle"),
  smoothSlider:  $("#smoothingSlider"),
  smoothValue:   $("#smoothingValue"),
  logScaleBtn:   $("#logScaleBtn"),
  dualAxisBtn:   $("#dualAxisBtn"),
  refreshBtn:    $("#refreshBtn"),
  tabBar:        $("#tabBar"),
  logStageSelect:  $("#logStageSelect"),
  loadLogsBtn:   $("#loadLogsBtn"),
  logOutput:     $("#logOutput"),
  inferMode:     $("#inferMode"),
  inferText:     $("#inferText"),
  inferCkpt:     $("#inferCkpt"),
  inferImage:    $("#inferImage"),
  inferAudioIn:  $("#inferAudioIn"),
  inferAudioOut: $("#inferAudioOut"),
  inferOCR:      $("#inferOCR"),
  inferBtn:      $("#inferBtn"),
  inferUnloadBtn: $("#inferUnloadBtn"),
  inferResult:   $("#inferResult"),
  testScript:    $("#testScript"),
  testSamples:   $("#testSamples"),
  testRunBtn:    $("#testRunBtn"),
  exportBtn:     $("#exportBtn"),
  testStatus:    $("#testStatus"),
};

// ── ECharts init ────────────────────────────────────────────
function initChart() {
  state.chart = echarts.init($("#chart"), null, { renderer: "canvas" });
  window.addEventListener("resize", () => state.chart && state.chart.resize());
}

// ── EMA smoothing ───────────────────────────────────────────
function emaSmooth(data, alpha) {
  if (alpha <= 0 || data.length === 0) return data;
  const result = [data[0]];
  for (let i = 1; i < data.length; i++) {
    result.push(alpha * result[i - 1] + (1 - alpha) * data[i]);
  }
  return result;
}

// ── Chart rendering ─────────────────────────────────────────
function updateChart() {
  if (!state.chart) return;
  const rows = getFilteredRows();
  const xKey = dom.xAxisSelect.value;
  const alpha = state.smoothing;

  // Group by (file + run_id + metric_name)
  const groups = new Map();
  for (const r of rows) {
    if (r.phase === "event") continue;
    const key = `${r._file || ""}|${r.run_id || ""}|${r.metric_name || ""}`;
    if (!groups.has(key)) groups.set(key, { metric: r.metric_name, points: [] });
    const x = Number(r[xKey] ?? 0);
    const y = Number(r.metric_value ?? 0);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      groups.get(key).points.push({ x, y });
    }
  }

  // Sort points
  for (const g of groups.values()) {
    g.points.sort((a, b) => a.x - b.x);
  }

  const palette = [
    "#8b5cf6", "#22d3ee", "#f59e0b", "#10b981", "#ef4444",
    "#f97316", "#60a5fa", "#a3e635", "#e879f7", "#fb923c",
  ];

  const series = [];
  const legendData = [];
  let idx = 0;
  const isLR = (name) => name === "lr" || name === "learning_rate";

  for (const [key, g] of groups) {
    const color = palette[idx % palette.length];
    const name = key.split("|").filter(Boolean).join(" / ");
    legendData.push(name);

    // Smoothed line
    const yVals = g.points.map(p => p.y);
    const smoothed = emaSmooth(yVals, alpha);

    series.push({
      name: name,
      type: "line",
      data: g.points.map((p, i) => [p.x, smoothed[i]]),
      smooth: false,
      symbol: "none",
      lineStyle: { width: 2, color },
      yAxisIndex: (state.dualAxis && isLR(g.metric)) ? 1 : 0,
      large: true,
      sampling: "lttb",
    });

    // Raw data (faint) if smoothing is active
    if (alpha > 0) {
      series.push({
        name: name + " (raw)",
        type: "line",
        data: g.points.map(p => [p.x, p.y]),
        smooth: false,
        symbol: "none",
        lineStyle: { width: 0.8, color, opacity: 0.25 },
        yAxisIndex: (state.dualAxis && isLR(g.metric)) ? 1 : 0,
        large: true,
        sampling: "lttb",
        showInLegend: false,
      });
    }
    idx++;
  }

  const option = {
    animation: false,
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(15, 18, 38, 0.9)",
      borderColor: "rgba(255,255,255,0.15)",
      textStyle: { color: "#e8ecf4", fontSize: 11, fontFamily: "Cascadia Code, monospace" },
      axisPointer: { type: "cross", lineStyle: { color: "rgba(255,255,255,0.2)" } },
    },
    legend: {
      data: legendData,
      textStyle: { color: "#9ca3bf", fontSize: 11 },
      top: 4,
      type: "scroll",
      pageTextStyle: { color: "#9ca3bf" },
    },
    grid: { left: 60, right: state.dualAxis ? 60 : 30, top: 40, bottom: 60 },
    toolbox: {
      feature: {
        saveAsImage: { title: "Save" },
        dataZoom: { title: { zoom: "Zoom", back: "Reset" } },
        restore: { title: "Restore" },
      },
      iconStyle: { borderColor: "#9ca3bf" },
      right: 10,
      top: 4,
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, filterMode: "none" },
      { type: "slider", xAxisIndex: 0, height: 20, bottom: 8,
        borderColor: "rgba(255,255,255,0.1)", fillerColor: "rgba(139,92,246,0.15)",
        handleStyle: { color: "#8b5cf6" }, textStyle: { color: "#9ca3bf" } },
    ],
    xAxis: {
      type: "value",
      name: xKey,
      nameTextStyle: { color: "#9ca3bf", fontSize: 11 },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
      axisLabel: { color: "#9ca3bf", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)" } },
    },
    yAxis: [
      {
        type: state.logScale ? "log" : "value",
        name: "Value",
        nameTextStyle: { color: "#9ca3bf", fontSize: 11 },
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
      },
      ...(state.dualAxis ? [{
        type: "value",
        name: "LR",
        nameTextStyle: { color: "#9ca3bf", fontSize: 11 },
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        splitLine: { show: false },
      }] : []),
    ],
    series,
  };

  state.chart.setOption(option, true);
}

// ── Filtering ───────────────────────────────────────────────
function getSelected(sel) {
  return new Set([...sel.selectedOptions].map(o => o.value));
}

function getFilteredRows() {
  const files = getSelected(dom.fileSelect);
  const runs = getSelected(dom.runSelect);
  const metrics = getSelected(dom.metricSelect);

  return state.allRows.filter(r => {
    if (r.phase === "event") return false;
    if (files.size && !files.has(r._file)) return false;
    if (runs.size && !runs.has(String(r.run_id || ""))) return false;
    if (metrics.size && !metrics.has(String(r.metric_name || ""))) return false;
    return true;
  });
}

function fillSelect(sel, values) {
  const prev = getSelected(sel);
  sel.innerHTML = values.map(v =>
    `<option${prev.has(v) ? " selected" : ""}>${v}</option>`
  ).join("");
}

// ── Chip-based filter system ────────────────────────────────
function buildChips(containerId, selectId, values) {
  const container = $(`#${containerId}`);
  const sel = $(`#${selectId}`);
  if (!container || !sel) return;

  // Preserve current selection
  const prevActive = new Set([...sel.selectedOptions].map(o => o.value));

  // Build hidden select
  sel.innerHTML = values.map(v =>
    `<option value="${v}"${prevActive.has(v) ? " selected" : ""}>${v}</option>`
  ).join("");

  // Build chips
  container.innerHTML = values.map(v => {
    const active = prevActive.has(v) ? " active" : "";
    const label = v.length > 24 ? v.slice(0, 22) + ".." : v;
    return `<span class="chip${active}" data-value="${v}" title="${v}">${label}</span>`;
  }).join("");

  // Click toggles
  container.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      chip.classList.toggle("active");
      // Sync hidden select
      const activeValues = new Set(
        [...container.querySelectorAll(".chip.active")].map(c => c.dataset.value)
      );
      [...sel.options].forEach(opt => { opt.selected = activeValues.has(opt.value); });
      updateChart();
      renderLatestValues();
    });
  });
}

function updateFilters() {
  const rows = state.allRows.filter(r => r.phase !== "event");
  const files = [...new Set(rows.map(r => r._file))].sort();
  const runs = [...new Set(rows.map(r => String(r.run_id || "")))].sort();
  const metrics = [...new Set(rows.map(r => String(r.metric_name || "")))].sort();

  buildChips("fileChips", "fileSelect", files);
  buildChips("runChips", "runSelect", runs);
  buildChips("metricChips", "metricSelect", metrics);

  // Also keep hidden selects in sync
  fillSelect(dom.fileSelect, files);
  fillSelect(dom.runSelect, runs);
  fillSelect(dom.metricSelect, metrics);
}

// ── Pipeline rendering ──────────────────────────────────────
function renderPipeline(stages) {
  if (!stages) { dom.pipeline.innerHTML = "<span style='color:var(--text-dim)'>Loading...</span>"; return; }

  dom.pipeline.innerHTML = Object.entries(stages).map(([id, s]) => {
    const statusClass = s.status;
    const meta = s.metadata;
    const stepInfo = meta ? `Step ${meta.step || 0}` : "";

    let actions = "";
    if (s.status === "idle" || s.status === "done") {
      actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Start</button>`;
      if (s.has_checkpoint) {
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete checkpoint files">Clear</button>`;
      }
    } else if (s.status === "running") {
      actions = `<button class="btn btn-sm btn-danger" onclick="stopStage('${id}')">Stop</button>`;
    } else if (s.status === "blocked") {
      actions = `<span style="font-size:10px;color:var(--warning)">Needs: ${s.blocked_by.join(", ")}</span>`;
    }

    return `
      <div class="stage-card">
        <div class="stage-id">Stage ${id}</div>
        <div class="stage-name">${s.name}</div>
        <div class="stage-status">
          <span class="status-dot ${statusClass}"></span>
          ${s.status} ${stepInfo ? "- " + stepInfo : ""}
        </div>
        <div class="stage-actions">${actions}</div>
      </div>
    `;
  }).join("");
}

// Global functions for onclick handlers
window.startStage = async function(stage) {
  const res = await api.post("/api/training/start", { stage });
  if (!res.ok) alert(res.error || "Failed to start");
  poll();
};

window.stopStage = async function(stage) {
  if (!confirm(`Stop training stage ${stage}?`)) return;
  const res = await api.post("/api/training/stop", { stage });
  if (!res.ok) alert(res.error || "Failed to stop");
  poll();
};

window.clearStage = async function(stage) {
  if (!confirm(`Delete ALL checkpoint files for stage ${stage}? This cannot be undone.`)) return;
  const res = await api.post("/api/training/clear", { stage });
  if (res.ok) {
    alert(`Cleared ${res.count} files from ${res.checkpoint_dir}`);
  } else {
    alert(res.error || "Failed to clear");
  }
  poll();
};

// ── Summary cards ───────────────────────────────────────────
function updateSummaryCards() {
  const rows = state.allRows.filter(r => r.phase !== "event");
  if (!rows.length) return;

  // Find latest training metrics
  const trainRows = rows.filter(r => r.phase === "train" && r.metric_name === "loss");
  const valRows = rows.filter(r => (r.phase === "val" || r.metric_name === "val_loss"));

  // Current step
  if (trainRows.length) {
    const latest = trainRows.reduce((a, b) => (a.step || 0) > (b.step || 0) ? a : b);
    setCard("cardStep", latest.step);
  }

  // Best val loss
  if (valRows.length) {
    const best = valRows.reduce((a, b) => (a.metric_value || Infinity) < (b.metric_value || Infinity) ? a : b);
    setCard("cardBestVal", best.metric_value?.toFixed(4));
  }

  // LR
  const withLR = trainRows.filter(r => r.lr != null);
  if (withLR.length) {
    const latest = withLR.reduce((a, b) => (a.step || 0) > (b.step || 0) ? a : b);
    setCard("cardLR", latest.lr?.toExponential(2));
  }

  // Throughput & ETA
  if (trainRows.length >= 2) {
    const sorted = [...trainRows].sort((a, b) => (a.step || 0) - (b.step || 0));
    const recent = sorted.slice(-20);
    if (recent.length >= 2) {
      const dt = (new Date(recent[recent.length - 1].timestamp) - new Date(recent[0].timestamp)) / 1000;
      const dSteps = (recent[recent.length - 1].step || 0) - (recent[0].step || 0);
      if (dt > 0 && dSteps > 0) {
        const stepsPerSec = dSteps / dt;
        setCard("cardThroughput", stepsPerSec.toFixed(1) + " s/s");

        // ETA: find max_steps from config
        const maxSteps = findMaxSteps();
        if (maxSteps) {
          const currentStep = sorted[sorted.length - 1].step || 0;
          const remaining = maxSteps - currentStep;
          if (remaining > 0) {
            const etaSec = remaining / stepsPerSec;
            setCard("cardETA", formatDuration(etaSec));
          } else {
            setCard("cardETA", "Done");
          }
        }
      }
    }
  }

  // GPU memory
  if (state.gpuData && state.gpuData.gpu) {
    const g = state.gpuData.gpu;
    const el = setCard("cardGPUMem", `${(g.memory_used_mb / 1024).toFixed(1)}/${(g.memory_total_mb / 1024).toFixed(0)} GB`);
  }
}

function findMaxSteps() {
  // Try to get from checkpoint configs
  if (state.checkpointData) {
    for (const ckpt of state.checkpointData) {
      if (ckpt.config && ckpt.config.max_steps) return ckpt.config.max_steps;
    }
  }
  return null;
}

function setCard(id, value) {
  const el = $(`#${id} .card-value`);
  if (el) el.textContent = value ?? "--";
  return el;
}

function formatDuration(sec) {
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

// ── GPU badge ───────────────────────────────────────────────
function updateGPUBadge() {
  if (!state.gpuData || !state.gpuData.gpu) {
    dom.gpuBadge.textContent = "GPU: N/A";
    dom.gpuBadge.className = "gpu-badge error";
    return;
  }
  const g = state.gpuData.gpu;
  dom.gpuBadge.textContent = `GPU: ${(g.memory_used_mb / 1024).toFixed(1)}/${(g.memory_total_mb / 1024).toFixed(0)} GB  ${g.utilization_percent}%  ${g.temperature_c}C`;
  dom.gpuBadge.className = "gpu-badge" + (g.memory_percent > 90 ? " warn" : g.memory_percent > 95 ? " error" : "");
}

// ── Tab: Latest Values ──────────────────────────────────────
function renderLatestValues() {
  const rows = getFilteredRows();
  const latest = new Map();
  for (const r of rows) {
    const key = `${r._file}|${r.metric_name}`;
    const prev = latest.get(key);
    if (!prev || (r.step || 0) >= (prev.step || 0)) latest.set(key, r);
  }

  const sorted = [...latest.values()].sort((a, b) =>
    String(a.metric_name).localeCompare(String(b.metric_name))
  );

  const tbody = $("#latestTable tbody");
  tbody.innerHTML = sorted.map(r => `
    <tr>
      <td style="color:var(--accent-2)">${r.metric_name || ""}</td>
      <td>${Number(r.metric_value ?? 0).toFixed(6)}</td>
      <td>${r.step ?? ""}</td>
      <td>${r.epoch ?? ""}</td>
      <td>${r.lr != null ? Number(r.lr).toExponential(2) : ""}</td>
      <td>${r.timestamp ? r.timestamp.replace("T", " ").slice(0, 19) : ""}</td>
    </tr>
  `).join("");
}

// ── Tab: Checkpoints ────────────────────────────────────────
function renderCheckpoints() {
  if (!state.checkpointData) return;
  const tbody = $("#checkpointTable tbody");
  tbody.innerHTML = state.checkpointData.map(c => {
    const meta = c.metadata || {};
    const cfg = c.config || {};
    const cfgStr = Object.entries(cfg).slice(0, 4).map(([k, v]) => `${k}=${v}`).join(", ");
    return `
      <tr>
        <td style="color:var(--accent)">${c.name}</td>
        <td>${meta.step ?? "--"}</td>
        <td>${meta.epoch ?? "--"}</td>
        <td>${c.size_mb}</td>
        <td style="font-size:10px">${cfgStr}</td>
        <td>${c.modified ? c.modified.replace("T", " ").slice(0, 19) : "--"}</td>
      </tr>
    `;
  }).join("");
}

// ── Tab: Hyperparameters ────────────────────────────────────
async function renderHyperparams() {
  const container = $("#hyperparamsContent");
  if (!state.checkpointData || !state.checkpointData.length) {
    container.innerHTML = "<span style='color:var(--text-dim)'>No checkpoints found</span>";
    return;
  }

  // Load full configs for each checkpoint that has one
  const cards = [];
  for (const ckpt of state.checkpointData) {
    if (!ckpt.has_config) continue;
    const res = await api.get(`/api/system/config/${ckpt.name}`);
    if (!res.ok) {
      // Try reading from checkpoint dir config
      cards.push({ name: ckpt.name, config: ckpt.config || {} });
      continue;
    }
    cards.push({ name: ckpt.name, config: res.config });
  }

  container.innerHTML = cards.map(c => {
    const rows = Object.entries(c.config).map(([k, v]) =>
      `<div class="hp-row"><span class="hp-key">${k}</span><span class="hp-val">${JSON.stringify(v)}</span></div>`
    ).join("");
    return `<div class="hp-card"><div class="hp-card-title">${c.name}</div>${rows}</div>`;
  }).join("");
}

// ── Tab: Events ─────────────────────────────────────────────
function renderEvents() {
  const events = state.allRows
    .filter(r => r.phase === "event")
    .sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""))
    .slice(0, 100);

  const tbody = $("#eventsTable tbody");
  tbody.innerHTML = events.map(r => `
    <tr>
      <td style="color:var(--warning)">${r.metric_name || ""}</td>
      <td>${r._file || ""}</td>
      <td>${r.run_id || ""}</td>
      <td>${r.step ?? ""}</td>
      <td>${r.epoch ?? ""}</td>
      <td>${r.timestamp ? r.timestamp.replace("T", " ").slice(0, 19) : ""}</td>
    </tr>
  `).join("");
}

// ── Polling ─────────────────────────────────────────────────
async function poll() {
  try {
    // Parallel fetches
    const [metricsRes, gpuRes, pipelineRes, ckptRes] = await Promise.all([
      api.get("/api/metrics/data?file=__all__"),
      api.get("/api/system/gpu"),
      api.get("/api/training/pipeline"),
      api.get("/api/system/checkpoints"),
    ]);

    // Metrics
    if (metricsRes.ok && metricsRes.file_data) {
      state.allRows = [];
      for (const [fname, rows] of Object.entries(metricsRes.file_data)) {
        for (const r of rows) {
          r._file = fname;
          state.allRows.push(r);
        }
      }
      updateFilters();
      updateChart();
      renderLatestValues();
      renderEvents();
    }

    // GPU
    if (gpuRes.ok) {
      state.gpuData = gpuRes;
      updateGPUBadge();
    }

    // Pipeline
    if (pipelineRes.ok) {
      state.pipelineData = pipelineRes.stages;
      renderPipeline(pipelineRes.stages);
    }

    // Checkpoints
    if (ckptRes.ok) {
      state.checkpointData = ckptRes.checkpoints;
      renderCheckpoints();
    }

    updateSummaryCards();
  } catch (e) {
    console.error("Poll error:", e);
  }
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(() => { if (state.live) poll(); }, state.pollMs);
}

// ── Tab switching ───────────────────────────────────────────
function setupTabs() {
  dom.tabBar.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    const tabId = btn.dataset.tab;

    $$(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");

    $$(".tab-content").forEach(tc => tc.classList.remove("active"));
    $(`#tab-${tabId}`).classList.add("active");

    // Load data for specific tabs
    if (tabId === "hyperparams") renderHyperparams();
    if (tabId === "configs" && $("#configSelect").options.length > 0) {
      $("#configLoadBtn").click();
    }
  });
}

// ── Collapsible panels ──────────────────────────────────────
function setupCollapsibles() {
  $$(".collapsible-header").forEach(header => {
    header.addEventListener("click", () => {
      const targetId = header.dataset.target;
      const body = $(`#${targetId}`);
      const isOpen = body.style.display !== "none";
      body.style.display = isOpen ? "none" : "block";
      header.classList.toggle("open", !isOpen);
    });
  });
}

// ── Event bindings ──────────────────────────────────────────
function setupControls() {
  // Live toggle
  dom.liveToggle.addEventListener("click", () => {
    state.live = !state.live;
    dom.liveToggle.classList.toggle("active", state.live);
    dom.liveToggle.textContent = state.live ? "LIVE" : "PAUSED";
  });

  // Smoothing
  dom.smoothSlider.addEventListener("input", () => {
    state.smoothing = parseFloat(dom.smoothSlider.value);
    dom.smoothValue.textContent = state.smoothing.toFixed(2);
    updateChart();
  });

  // Log scale
  dom.logScaleBtn.addEventListener("click", () => {
    state.logScale = !state.logScale;
    dom.logScaleBtn.style.borderColor = state.logScale ? "var(--accent)" : "";
    updateChart();
  });

  // Dual axis
  dom.dualAxisBtn.addEventListener("click", () => {
    state.dualAxis = !state.dualAxis;
    dom.dualAxisBtn.style.borderColor = state.dualAxis ? "var(--accent)" : "";
    updateChart();
  });

  // Refresh
  dom.refreshBtn.addEventListener("click", poll);

  // X-axis toggle
  dom.xAxisToggle.addEventListener("click", () => {
    const cur = dom.xAxisSelect.value;
    const next = cur === "step" ? "epoch" : "step";
    dom.xAxisSelect.value = next;
    dom.xAxisToggle.textContent = `X: ${next}`;
    dom.xAxisToggle.style.borderColor = next === "epoch" ? "var(--accent)" : "";
    updateChart();
  });

  // Filter changes
  [dom.fileSelect, dom.runSelect, dom.metricSelect, dom.xAxisSelect].forEach(el => {
    el.addEventListener("change", () => { updateChart(); renderLatestValues(); });
  });

  // Config editor
  setupConfigEditor();

  // Logs
  dom.loadLogsBtn.addEventListener("click", async () => {
    const stage = dom.logStageSelect.value;
    const res = await api.get(`/api/training/logs/${stage}`);
    dom.logOutput.textContent = res.ok ? res.lines.join("\n") : "Failed to load logs";
  });

  // Inference — mode changes checkpoint list
  async function updateInferCkptList() {
    const mode = dom.inferMode.value;
    const label = $("#inferCkptLabel");
    if (mode === "normal") {
      label.style.display = "";
      const res = await api.get("/api/system/checkpoints");
      if (res.ok) {
        dom.inferCkpt.innerHTML = res.checkpoints
          .filter(c => c.has_model)
          .map(c => `<option value="${c.path}">${c.name}</option>`)
          .join("");
      }
    } else {
      // Standalone / HuggingFace always use export/
      label.style.display = "none";
      dom.inferCkpt.innerHTML = '<option value="export">export/</option>';
    }
  }
  dom.inferMode.addEventListener("change", updateInferCkptList);
  updateInferCkptList();

  // Inference — 3 modes
  dom.inferBtn.addEventListener("click", async () => {
    const mode = dom.inferMode.value;
    const text = dom.inferText.value.trim();
    const source = dom.inferCkpt.value;
    const imagePath = dom.inferImage?.value.trim() || null;
    const audioIn = dom.inferAudioIn?.value.trim() || null;
    const audioOut = dom.inferAudioOut?.value.trim() || null;
    const useOCR = dom.inferOCR?.checked || false;

    if (!text && !imagePath && !audioIn) {
      dom.inferResult.textContent = "Provide text, image, or audio input.";
      return;
    }

    dom.inferResult.textContent = `Running ${mode} inference...`;
    dom.inferResult.className = "infer-result loading";

    let res;
    if (mode === "normal") {
      res = await api.post("/api/inference/chat", {
        text, ckpt_dir: source, image_path: imagePath,
        audio_in: audioIn, audio_out: audioOut, use_ocr: useOCR,
      });
    } else if (mode === "standalone") {
      res = await api.post("/api/inference/standalone", {
        text, model_dir: source, max_tokens: 64,
      });
    } else {
      res = await api.post("/api/inference/huggingface", {
        text, model_dir: source, image_path: imagePath, audio_path: audioIn,
        multimodal: !!(imagePath || audioIn), max_tokens: 32,
      });
    }

    dom.inferResult.className = "infer-result";
    if (res.ok) {
      const parts = [];
      if (res.response) parts.push(res.response);
      if (res.image_response) parts.push(`[Image] ${res.image_response}`);
      if (res.audio_response) parts.push(`[Audio] ${res.audio_response}`);
      if (res.audio_out) parts.push(`Audio saved: ${res.audio_out}`);
      parts.push(`\n[${res.mode || mode} | ${res.elapsed_ms}ms]`);
      dom.inferResult.textContent = parts.join("\n");
    } else {
      dom.inferResult.textContent = `Error: ${res.error || "Unknown error"}`;
    }
  });

  // Unload model
  dom.inferUnloadBtn.addEventListener("click", async () => {
    const res = await api.post("/api/inference/unload", {});
    dom.inferResult.textContent = res.ok ? "Model unloaded. VRAM freed." : "Failed to unload.";
  });

  // Testing
  dom.testRunBtn.addEventListener("click", async () => {
    const script = dom.testScript.value;
    const samples = parseInt(dom.testSamples.value) || 50;
    dom.testStatus.textContent = `Starting ${script}...`;
    const res = await api.post("/api/testing/run", { script, num_samples: samples });
    dom.testStatus.textContent = res.ok
      ? `Running: ${script} (PID ${res.pid})`
      : `Error: ${res.error || "Failed"}`;
  });

  // Export
  dom.exportBtn.addEventListener("click", async () => {
    dom.testStatus.textContent = "Starting export...";
    const res = await api.post("/api/export/run", {});
    dom.testStatus.textContent = res.ok
      ? `Export running (PID ${res.pid})`
      : `Error: ${res.error || "Failed"}`;
  });
}

// ── HP Tuning ───────────────────────────────────────────────
async function setupTuning() {
  const stageSel = $("#tuneStage");
  const paramsDiv = $("#tuneParams");
  const resultsDiv = $("#tuneResults");

  // Load search spaces
  const spacesRes = await api.get("/api/tuning/spaces");
  const spaces = spacesRes.ok ? spacesRes.spaces : {};

  // Show params when stage changes
  async function showParams() {
    const stage = stageSel.value;
    const space = spaces[stage];
    if (!space) { paramsDiv.innerHTML = ""; return; }

    paramsDiv.innerHTML = space.params.map(p => {
      let range = "";
      if (p.type === "float_log" || p.type === "float") range = `${p.low} — ${p.high}`;
      else if (p.type === "int") range = `${p.low} — ${p.high}`;
      else if (p.type === "categorical") range = p.choices.map(c => JSON.stringify(c)).join(", ");
      return `<div class="hp-card" style="padding:8px 10px">
        <span class="hp-key">${p.name}</span>
        <span class="hp-val" style="font-size:10px">${p.type}: ${range}</span>
      </div>`;
    }).join("");

    // Load existing results
    const resData = await api.get(`/api/tuning/results/${stage}`);
    if (resData.ok && resData.results) {
      const r = resData.results;
      if (r.error) {
        resultsDiv.textContent = `Error: ${r.error}`;
      } else if (r.best_trial) {
        resultsDiv.innerHTML = `<strong>Best:</strong> val_loss=${r.best_trial.value?.toFixed(4)} (trial #${r.best_trial.number}) | ${r.n_trials} trials total`;
        // Fill trials table
        const tbody = $("#tuneTrialsTable tbody");
        const trials = (r.trials || []).sort((a, b) => (a.value || 999) - (b.value || 999));
        tbody.innerHTML = trials.slice(0, 20).map(t => `<tr>
          <td>${t.number}</td>
          <td style="color:${t.state === 'COMPLETE' ? 'var(--success)' : 'var(--text-dim)'}">${t.value?.toFixed(4) ?? "--"}</td>
          <td>${t.state}</td>
          <td>${t.duration_seconds ? t.duration_seconds.toFixed(0) + "s" : "--"}</td>
          <td style="font-size:9px">${Object.entries(t.params || {}).map(([k,v]) => k + "=" + (typeof v === 'number' ? v.toPrecision(3) : v)).join(", ")}</td>
        </tr>`).join("");
      } else {
        resultsDiv.textContent = "No results yet. Start a tuning run.";
      }
    } else {
      resultsDiv.textContent = "No tuning DB found for this stage.";
    }
  }

  stageSel.addEventListener("change", showParams);
  showParams();

  // Start tuning
  $("#tuneStartBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    const nTrials = parseInt($("#tuneTrials").value) || 30;
    const maxSteps = parseInt($("#tuneSteps").value) || 2000;
    resultsDiv.textContent = `Starting tuning for Stage ${stage}...`;
    const res = await api.post("/api/tuning/start", { stage, n_trials: nTrials, max_steps: maxSteps });
    resultsDiv.textContent = res.ok
      ? `Tuning running: Stage ${stage}, ${nTrials} trials (PID ${res.pid})`
      : `Error: ${res.error || "Failed"}`;
  });

  // Stop tuning
  $("#tuneStopBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    const res = await api.post("/api/tuning/stop", { stage });
    resultsDiv.textContent = res.ok ? `Stopped tuning for Stage ${stage}` : `Error: ${res.error}`;
  });
}

// ── Config Editor ───────────────────────────────────────────
async function setupConfigEditor() {
  const configSel = $("#configSelect");
  const ckptSel = $("#configCheckpointSelect");
  const editor = $("#configEditor");
  const ckptView = $("#configCheckpointView");
  const statusEl = $("#configStatus");

  // Load config list
  const res = await api.get("/api/system/configs");
  if (res.ok) {
    configSel.innerHTML = res.configs.map(c => `<option value="${c}">${c}</option>`).join("");
  }

  // Load checkpoint list for comparison
  const ckptRes = await api.get("/api/system/checkpoints");
  if (ckptRes.ok) {
    ckptSel.innerHTML = '<option value="">-- none --</option>' +
      ckptRes.checkpoints.filter(c => c.has_config).map(c =>
        `<option value="${c.name}">${c.name}</option>`
      ).join("");
  }

  // Load button
  $("#configLoadBtn").addEventListener("click", async () => {
    const name = configSel.value;
    if (!name) return;

    const r = await api.get(`/api/system/config/${name}`);
    if (r.ok) {
      editor.value = JSON.stringify(r.config, null, 2);
      statusEl.textContent = `Loaded: ${name}`;
    } else {
      statusEl.textContent = `Error: ${r.error}`;
    }

    // Load checkpoint config if selected
    const ckptName = ckptSel.value;
    if (ckptName) {
      const cr = await api.get(`/api/system/checkpoint-config/${ckptName}`);
      if (cr.ok) {
        ckptView.value = JSON.stringify(cr.config, null, 2);
      } else {
        ckptView.value = "(not found)";
      }
    } else {
      ckptView.value = "";
    }
  });

  // Save button
  $("#configSaveBtn").addEventListener("click", async () => {
    const name = configSel.value;
    if (!name) return;

    let config;
    try {
      config = JSON.parse(editor.value);
    } catch (e) {
      statusEl.textContent = `Invalid JSON: ${e.message}`;
      return;
    }

    const r = await api.post(`/api/system/config/${name}`, { config });
    statusEl.textContent = r.ok ? `Saved: ${name}` : `Error: ${r.error}`;
  });

  // Auto-load first config
  if (configSel.options.length > 0) {
    $("#configLoadBtn").click();
  }
}

// ── Init ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  setupTabs();
  setupCollapsibles();
  setupControls();
  setupTuning();
  poll();
  startPolling();
});
