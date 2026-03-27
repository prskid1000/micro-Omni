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

// ── Toast notifications ─────────────────────────────────────
function showToast(message, type = "info", duration = 5000) {
  const stack = $("#toastStack");
  if (!stack) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icon = type === "success" ? "&#10003;" : type === "error" ? "&#10007;" : type === "warning" ? "&#9888;" : "&#8505;";
  toast.innerHTML = `<span>${icon}</span> ${message}`;
  stack.appendChild(toast);
  if (duration > 0) setTimeout(() => toast.remove(), duration);
  // Max 3 visible
  while (stack.children.length > 3) stack.children[0].remove();
}

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
  // Resize ALL echarts instances on window resize
  window.addEventListener("resize", () => {
    document.querySelectorAll("div").forEach(el => {
      const inst = echarts.getInstanceByDom(el);
      if (inst) inst.resize();
    });
  });
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
    const proc = s.process;
    const maxSteps = meta?.max_steps || null;

    // Build status line
    let statusLine = s.status;
    if (s.status === "running" && proc) {
      statusLine = `running`;
    }
    if (stepInfo && s.status !== "idle" && s.status !== "blocked") {
      statusLine += ` - ${stepInfo}`;
    }

    // Build action buttons based on state
    let actions = "";
    switch (s.status) {
      case "idle":
        actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Start</button>`;
        break;
      case "running":
        actions = `<button class="btn btn-sm btn-danger" onclick="stopStage('${id}')">Stop</button>`;
        break;
      case "stopped":
        actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Resume</button>`;
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete and restart">Clear</button>`;
        break;
      case "failed":
        actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Retry</button>`;
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete and restart">Clear</button>`;
        break;
      case "done":
        actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Retrain</button>`;
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete checkpoint files">Clear</button>`;
        break;
      case "blocked":
        actions = `<span style="font-size:10px;color:var(--warning)">Needs: ${s.blocked_by.join(", ")}</span>`;
        break;
    }

    return `
      <div class="stage-card">
        <div class="stage-id">Stage ${id}</div>
        <div class="stage-name">${s.name}</div>
        <div class="stage-status">
          <span class="status-dot ${statusClass}"></span>
          ${statusLine}
        </div>
        <div class="stage-actions">${actions}</div>
      </div>
    `;
  }).join("");
}

// Global functions for onclick handlers
window.startStage = async function(stage) {
  const res = await api.post("/api/training/start", { stage });
  if (res.ok) {
    showToast(`Stage ${stage} started (PID ${res.pid})`, "success");
  } else {
    showToast(res.error || "Failed to start", "error");
  }
  poll();
};

window.stopStage = async function(stage) {
  if (!confirm(`Stop training stage ${stage}?`)) return;
  const res = await api.post("/api/training/stop", { stage });
  if (res.ok) {
    showToast(`Stage ${stage} stopped`, "warning");
  } else {
    showToast(res.error || "Failed to stop", "error");
  }
  poll();
};

window.clearStage = async function(stage) {
  if (!confirm(`Delete ALL checkpoint files for stage ${stage}? This cannot be undone.`)) return;
  const res = await api.post("/api/training/clear", { stage });
  if (res.ok) {
    showToast(`Cleared ${res.count} files from ${res.checkpoint_dir}`, "success");
  } else {
    showToast(res.error || "Failed to clear", "error");
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
    const [metricsRes, gpuRes, pipelineRes, ckptRes, testStatusRes, tuneStatusRes, exportStatusRes] = await Promise.all([
      api.get("/api/metrics/data?file=__all__"),
      api.get("/api/system/gpu"),
      api.get("/api/training/pipeline"),
      api.get("/api/system/checkpoints"),
      api.get("/api/testing/status"),
      api.get("/api/tuning/status"),
      api.get("/api/export/status"),
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

    // Pipeline — detect state changes and notify
    if (pipelineRes.ok) {
      const oldStages = state.pipelineData;
      const newStages = pipelineRes.stages;

      if (oldStages) {
        for (const [id, s] of Object.entries(newStages)) {
          const old = oldStages[id];
          if (!old) continue;
          // Training just completed
          if (old.status === "running" && s.status === "done") {
            showToast(`Stage ${id} (${s.name}) completed!`, "success", 10000);
            document.title = `[Done] micro-Omni Dashboard`;
            setTimeout(() => { document.title = "micro-Omni Dashboard"; }, 30000);
          }
          // Training failed
          if (old.status === "running" && s.status === "failed") {
            showToast(`Stage ${id} (${s.name}) failed! Check logs.`, "error", 15000);
            document.title = `[Failed] micro-Omni Dashboard`;
          }
          // Training stopped by user
          if (old.status === "running" && s.status === "stopped") {
            showToast(`Stage ${id} (${s.name}) stopped.`, "warning", 8000);
          }
        }
      }

      state.pipelineData = newStages;
      renderPipeline(newStages);
    }

    // Checkpoints
    if (ckptRes.ok) {
      state.checkpointData = ckptRes.checkpoints;
      renderCheckpoints();
    }

    // Restore testing/export status from server process state
    if (testStatusRes.ok || exportStatusRes.ok) {
      const el = $("#testStatus");
      if (el) {
        const allProcs = [
          ...Object.values((testStatusRes.ok && testStatusRes.processes) || {}),
          ...Object.values((exportStatusRes.ok && exportStatusRes.processes) || {}),
        ];
        // Show most recent process status
        const sorted = allProcs.sort((a, b) => (b.start_time || "").localeCompare(a.start_time || ""));
        const latest = sorted[0];
        if (latest) {
          const elapsed = latest.elapsed_seconds ? ` (${Math.round(latest.elapsed_seconds)}s)` : "";
          if (latest.status === "running") {
            el.textContent = `Running: ${latest.stage} (PID ${latest.pid})${elapsed}`;
          } else if (latest.status === "completed") {
            el.textContent = `Completed: ${latest.stage}${elapsed}`;
          } else if (latest.status === "failed") {
            el.innerHTML = `<span style="color:var(--danger)">Failed: ${latest.stage}${elapsed}</span>`;
          } else if (latest.status === "stopped") {
            el.textContent = `Stopped: ${latest.stage}${elapsed}`;
          }
        }
      }
    }

    // If tuning is running, keep its progress indicator alive
    if (tuneStatusRes.ok && tuneStatusRes.processes) {
      const tuneRunning = Object.values(tuneStatusRes.processes).some(p => p.status === "running");
      if (tuneRunning && typeof window._pollTuningProgress === "function") {
        window._pollTuningProgress();
      }
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
    if (tabId === "inference" && typeof window._updateInferCkptList === "function") window._updateInferCkptList();

    // Resize all ECharts instances in the newly visible tab
    // (charts initialized while tab was hidden get 0 width)
    setTimeout(() => {
      const tabEl = $(`#tab-${tabId}`);
      if (tabEl) {
        tabEl.querySelectorAll("div").forEach(el => {
          const inst = echarts.getInstanceByDom(el);
          if (inst) inst.resize();
        });
      }
    }, 100);
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
  window._updateInferCkptList = updateInferCkptList;
  dom.inferMode.addEventListener("change", updateInferCkptList);
  updateInferCkptList();

  // ── Test results chart + table ─────────────────────────
  let testChart = null;
  $("#testLoadResultsBtn").addEventListener("click", async () => {
    const script = dom.testScript.value;
    const res = await api.get(`/api/testing/results/${script}`);
    if (!res.ok || !res.results.length) {
      dom.testStatus.textContent = "No results found for " + script;
      return;
    }

    // Fill table
    const tbody = $("#testResultsTable tbody");
    const rows = res.results.filter(r => r.phase !== "event");
    tbody.innerHTML = rows.map(r => `<tr>
      <td style="color:var(--accent-2)">${r.metric_name || ""}</td>
      <td>${r.metric_value != null ? Number(r.metric_value).toFixed(4) : "--"}</td>
      <td>${r.phase || ""}</td>
      <td>${r.timestamp ? r.timestamp.replace("T", " ").slice(0, 19) : ""}</td>
    </tr>`).join("");

    // Render bar chart of metrics
    if (!testChart) testChart = echarts.init($("#testChart"));
    const metrics = {};
    for (const r of rows) {
      if (r.metric_name && r.metric_value != null) metrics[r.metric_name] = r.metric_value;
    }
    const names = Object.keys(metrics);
    const values = Object.values(metrics);

    testChart.setOption({
      animation: false,
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", backgroundColor: "rgba(15,18,38,0.9)", borderColor: "rgba(255,255,255,0.15)", textStyle: { color: "#e8ecf4", fontSize: 11 } },
      grid: { left: 60, right: 20, top: 20, bottom: 40 },
      xAxis: { type: "category", data: names, axisLabel: { color: "#9ca3bf", fontSize: 10, rotate: 30 }, axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } } },
      yAxis: { type: "value", axisLabel: { color: "#9ca3bf", fontSize: 10 }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } } },
      series: [{ type: "bar", data: values.map(v => ({ value: v, itemStyle: { color: v < 1 ? "#10b981" : v < 5 ? "#22d3ee" : "#f59e0b" } })), barWidth: "60%" }],
    }, true);

    dom.testStatus.textContent = `Loaded ${rows.length} metrics for ${script}`;

    // Also load cross-model comparison
    await loadTestComparison();
  });

  // Cross-model comparison: load all test results and compare key metrics
  let testCompareChartInstance = null;
  async function loadTestComparison() {
    const allScripts = ["test_thinker", "test_audio_enc", "test_vision", "test_talker", "test_vocoder", "test_ocr", "test_sft"];
    const allResults = {};

    for (const script of allScripts) {
      const res = await api.get(`/api/testing/results/${script}`);
      if (res.ok && res.results && res.results.length) {
        const metrics = {};
        for (const r of res.results) {
          if (r.phase !== "event" && r.metric_name && r.metric_value != null) {
            metrics[r.metric_name] = r.metric_value;
          }
        }
        if (Object.keys(metrics).length) allResults[script] = metrics;
      }
    }

    if (Object.keys(allResults).length === 0) return;

    // Find common metric names across models (e.g. "perplexity", "accuracy")
    const allMetricNames = new Set();
    for (const m of Object.values(allResults)) { for (const k of Object.keys(m)) allMetricNames.add(k); }

    // Build grouped bar chart: each model is a group, each metric is a bar
    const models = Object.keys(allResults);
    const metricNames = [...allMetricNames].sort();
    const palette = ["#8b5cf6", "#22d3ee", "#f59e0b", "#10b981", "#ef4444", "#f97316", "#60a5fa"];

    const series = metricNames.map((metric, i) => ({
      name: metric,
      type: "bar",
      data: models.map(model => allResults[model][metric] ?? null),
      itemStyle: { color: palette[i % palette.length] },
    }));

    if (!testCompareChartInstance) testCompareChartInstance = echarts.init($("#testCompareChart"));
    testCompareChartInstance.setOption({
      animation: false,
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", backgroundColor: "rgba(15,18,38,0.9)", borderColor: "rgba(255,255,255,0.15)", textStyle: { color: "#e8ecf4", fontSize: 11 } },
      legend: { data: metricNames, textStyle: { color: "#9ca3bf", fontSize: 10 }, top: 4, type: "scroll" },
      grid: { left: 60, right: 20, top: 40, bottom: 50 },
      xAxis: {
        type: "category",
        data: models.map(m => m.replace("test_", "")),
        axisLabel: { color: "#9ca3bf", fontSize: 10, rotate: 20 },
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
      },
      series,
    }, true);
  }

  // ── Helper: disable button during async operation ──────
  async function withDisable(btn, fn) {
    const origText = btn.textContent;
    btn.disabled = true;
    btn.style.opacity = "0.5";
    try { await fn(); }
    finally { btn.disabled = false; btn.style.opacity = ""; }
  }

  // Inference — 3 modes
  dom.inferBtn.addEventListener("click", () => withDisable(dom.inferBtn, async () => {
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
    dom.inferBtn.textContent = "Running...";

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

    dom.inferBtn.textContent = "Run Inference";
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
  }));

  // Unload model
  dom.inferUnloadBtn.addEventListener("click", () => withDisable(dom.inferUnloadBtn, async () => {
    dom.inferUnloadBtn.textContent = "Unloading...";
    const res = await api.post("/api/inference/unload", {});
    dom.inferResult.textContent = res.ok ? "Model unloaded. VRAM freed." : "Failed to unload.";
    dom.inferUnloadBtn.textContent = "Unload Model";
  }));

  // Testing
  dom.testRunBtn.addEventListener("click", () => withDisable(dom.testRunBtn, async () => {
    const script = dom.testScript.value;
    const samples = parseInt(dom.testSamples.value) || 50;
    dom.testRunBtn.textContent = "Starting...";
    dom.testStatus.textContent = `Starting ${script}...`;
    const res = await api.post("/api/testing/run", { script, num_samples: samples });
    dom.testRunBtn.textContent = "Run Test";
    dom.testStatus.textContent = res.ok
      ? `Running: ${script} (PID ${res.pid})`
      : `Error: ${res.error || "Failed"}`;
  }));

  // Export
  dom.exportBtn.addEventListener("click", () => withDisable(dom.exportBtn, async () => {
    dom.exportBtn.textContent = "Exporting...";
    dom.testStatus.textContent = "Starting export...";
    const res = await api.post("/api/export/run", {});
    dom.exportBtn.textContent = "Export Model";
    dom.testStatus.textContent = res.ok
      ? `Export running (PID ${res.pid})`
      : `Error: ${res.error || "Failed"}`;
  }));
}

// ── HP Tuning ───────────────────────────────────────────────
async function setupTuning() {
  const stageSel = $("#tuneStage");
  const paramsDiv = $("#tuneParams");
  const resultsDiv = $("#tuneResults");
  const progressDiv = $("#tuneProgress");

  let tuneSpaces = {};
  let tunePollTimer = null;
  let tuneChart = null;
  let tuneTrainChart = null;
  const tuneSliceInstances = {};  // param name -> echarts instance

  // Load search spaces
  const spacesRes = await api.get("/api/tuning/spaces");
  tuneSpaces = spacesRes.ok ? spacesRes.spaces : {};

  // Render search space params
  function renderParams() {
    const stage = stageSel.value;
    const space = tuneSpaces[stage];
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
  }

  // Poll tuning progress + results
  async function pollTuningProgress() {
    const stage = stageSel.value;

    // Check if tuning is running
    const statusRes = await api.get("/api/tuning/status");
    const isRunning = statusRes.ok && Object.values(statusRes.processes || {}).some(
      p => p.status === "running" && p.stage === `tune_${stage}`
    );

    // Load results from DB (updates as trials complete)
    const resData = await api.get(`/api/tuning/results/${stage}`);

    if (resData.ok && resData.results && !resData.results.error) {
      const r = resData.results;
      const completedTrials = (r.trials || []).filter(t => t.state === "COMPLETE");
      const prunedTrials = (r.trials || []).filter(t => t.state === "PRUNED");
      const failedTrials = (r.trials || []).filter(t => t.state === "FAIL");

      // Progress bar
      if (isRunning) {
        const total = r.n_trials;
        const done = completedTrials.length + prunedTrials.length + failedTrials.length;
        const pct = total > 0 ? Math.round(done / total * 100) : 0;
        progressDiv.innerHTML = `
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span class="status-dot running"></span>
            <strong>Running:</strong> Trial ${done}/${total} (${pct}%)
            <span style="color:var(--success)">${completedTrials.length} complete</span>
            <span style="color:var(--warning)">${prunedTrials.length} pruned</span>
            ${failedTrials.length ? `<span style="color:var(--danger)">${failedTrials.length} failed</span>` : ""}
          </div>
          <div style="height:4px;background:var(--bg-surface);border-radius:2px;overflow:hidden">
            <div style="height:100%;width:${pct}%;background:linear-gradient(90deg,var(--accent),var(--accent-2));transition:width 0.3s"></div>
          </div>
        `;
      } else {
        progressDiv.innerHTML = r.n_trials > 0
          ? `<span style="color:var(--text-dim)">Completed: ${r.n_trials} trials</span>`
          : "";
      }

      // Best result
      if (r.best_trial) {
        const bestParams = Object.entries(r.best_trial.params || {})
          .map(([k, v]) => `<span class="hp-key">${k}</span>=<span class="hp-val">${typeof v === "number" ? v.toPrecision(4) : v}</span>`)
          .join(", ");
        resultsDiv.innerHTML = `
          <div style="margin-bottom:6px"><strong style="color:var(--success)">Best: val_loss = ${r.best_trial.value?.toFixed(4)}</strong> (trial #${r.best_trial.number})</div>
          <div style="font-size:11px">${bestParams}</div>
        `;
      } else {
        resultsDiv.textContent = isRunning ? "Waiting for first trial to complete..." : "No results yet.";
      }

      // Trials table (sorted by val_loss)
      const tbody = $("#tuneTrialsTable tbody");
      const allTrials = (r.trials || []).sort((a, b) => (a.value ?? 999) - (b.value ?? 999));
      tbody.innerHTML = allTrials.slice(0, 30).map((t, i) => {
        const stateColor = t.state === "COMPLETE" ? "var(--success)" : t.state === "PRUNED" ? "var(--warning)" : "var(--danger)";
        const isBest = i === 0 && t.state === "COMPLETE";
        return `<tr${isBest ? ' style="background:rgba(16,185,129,0.08)"' : ""}>
          <td>${isBest ? "★ " : ""}${t.number}</td>
          <td style="color:${stateColor}">${t.value?.toFixed(4) ?? "--"}</td>
          <td style="color:${stateColor}">${t.state}</td>
          <td>${t.duration_seconds ? t.duration_seconds.toFixed(0) + "s" : "--"}</td>
          <td style="font-size:9px;max-width:400px;overflow:hidden;text-overflow:ellipsis">${
            Object.entries(t.params || {}).map(([k, v]) => k + "=" + (typeof v === "number" ? v.toPrecision(3) : v)).join(", ")
          }</td>
        </tr>`;
      }).join("");

      // Render optimization history chart (val_loss vs trial)
      if (!tuneChart) tuneChart = echarts.init($("#tuneChart"));
      const chartTrials = (r.trials || []).filter(t => t.state === "COMPLETE" && t.value != null)
        .sort((a, b) => a.number - b.number);
      if (chartTrials.length > 0) {
        // Running best line
        let runningBest = [];
        let best = Infinity;
        for (const t of chartTrials) { best = Math.min(best, t.value); runningBest.push(best); }

        tuneChart.setOption({
          animation: false,
          backgroundColor: "transparent",
          tooltip: {
            trigger: "axis",
            backgroundColor: "rgba(15,18,38,0.9)",
            borderColor: "rgba(255,255,255,0.15)",
            textStyle: { color: "#e8ecf4", fontSize: 11 },
          },
          legend: { data: ["Val Loss", "Best So Far"], textStyle: { color: "#9ca3bf", fontSize: 11 }, top: 4 },
          grid: { left: 60, right: 20, top: 35, bottom: 30 },
          xAxis: {
            type: "value", name: "Trial", nameTextStyle: { color: "#9ca3bf" },
            axisLabel: { color: "#9ca3bf", fontSize: 10 },
            axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
          },
          yAxis: {
            type: "value", name: "Val Loss", nameTextStyle: { color: "#9ca3bf" },
            axisLabel: { color: "#9ca3bf", fontSize: 10 },
            splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
          },
          series: [
            {
              name: "Val Loss", type: "scatter", symbolSize: 8,
              data: chartTrials.map(t => [t.number, t.value]),
              itemStyle: { color: "#8b5cf6" },
            },
            {
              name: "Best So Far", type: "line", smooth: true, symbol: "none",
              data: chartTrials.map((t, i) => [t.number, runningBest[i]]),
              lineStyle: { color: "#10b981", width: 2 },
              areaStyle: { color: "rgba(16, 185, 129, 0.08)" },
            },
          ],
        }, true);
      }

      // ── Slice plots: each param vs val_loss ──────────────
      const sliceContainer = $("#tuneSliceCharts");
      if (chartTrials.length >= 3) {
        // Collect all param names from completed trials
        const paramNames = new Set();
        for (const t of chartTrials) { for (const k of Object.keys(t.params || {})) paramNames.add(k); }

        // Create/update a mini chart for each param
        const sortedParams = [...paramNames].sort();

        // Build container divs if needed
        if (sliceContainer.children.length !== sortedParams.length) {
          sliceContainer.innerHTML = sortedParams.map(p =>
            `<div class="tune-slice-chart" id="slice-${p.replace(/[^a-zA-Z0-9_]/g, '_')}"></div>`
          ).join("");
          // Dispose old instances
          for (const k of Object.keys(tuneSliceInstances)) {
            try { tuneSliceInstances[k].dispose(); } catch {}
            delete tuneSliceInstances[k];
          }
        }

        for (const paramName of sortedParams) {
          const elId = `slice-${paramName.replace(/[^a-zA-Z0-9_]/g, '_')}`;
          const el = $(`#${elId}`);
          if (!el) continue;

          if (!tuneSliceInstances[paramName]) {
            tuneSliceInstances[paramName] = echarts.init(el);
          }
          const chart = tuneSliceInstances[paramName];

          const points = chartTrials
            .filter(t => t.params && t.params[paramName] !== undefined)
            .map(t => {
              let x = t.params[paramName];
              if (typeof x === "boolean") x = x ? 1 : 0;
              if (typeof x === "string") x = x === "true" ? 1 : x === "false" ? 0 : 0;
              return [x, t.value];
            })
            .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));

          const isCategorical = points.length > 0 && new Set(points.map(p => p[0])).size <= 6;

          chart.setOption({
            animation: false,
            backgroundColor: "transparent",
            title: { text: paramName, textStyle: { color: "#9ca3bf", fontSize: 11, fontWeight: 600 }, left: 8, top: 4 },
            tooltip: {
              trigger: "item",
              backgroundColor: "rgba(15,18,38,0.9)",
              borderColor: "rgba(255,255,255,0.15)",
              textStyle: { color: "#e8ecf4", fontSize: 10 },
              formatter: p => `${paramName}: ${p.value[0]}<br>val_loss: ${p.value[1].toFixed(4)}`,
            },
            grid: { left: 50, right: 12, top: 30, bottom: 28 },
            xAxis: {
              type: "value",
              axisLabel: { color: "#7a82a0", fontSize: 9 },
              axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
              splitLine: { show: false },
            },
            yAxis: {
              type: "value", name: "val_loss",
              nameTextStyle: { color: "#7a82a0", fontSize: 9 },
              axisLabel: { color: "#7a82a0", fontSize: 9 },
              splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)" } },
            },
            series: [{
              type: "scatter",
              symbolSize: 7,
              data: points,
              itemStyle: {
                color: (p) => {
                  // Color by rank: best = green, worst = red
                  const vals = points.map(q => q[1]).sort((a, b) => a - b);
                  const rank = vals.indexOf(p.value[1]) / Math.max(vals.length - 1, 1);
                  return rank < 0.33 ? "#10b981" : rank < 0.66 ? "#f59e0b" : "#ef4444";
                },
              },
            }],
          }, true);
        }
      } else {
        sliceContainer.innerHTML = '<span style="color:var(--text-dim);font-size:11px">Need 3+ completed trials for slice plots</span>';
      }

      // ── Training curves from tuning trial metrics ─────────
      await renderTuneTrainCurves(stageSel.value);

    } else {
      progressDiv.innerHTML = isRunning
        ? '<span class="status-dot running"></span> Tuning starting...'
        : "";
      resultsDiv.textContent = "No tuning DB found for this stage. Start a tuning run.";
      $("#tuneTrialsTable tbody").innerHTML = "";
      $("#tuneSliceCharts").innerHTML = "";
    }

    // Auto-poll while running
    if (isRunning && !tunePollTimer) {
      tunePollTimer = setInterval(pollTuningProgress, 8000);
    } else if (!isRunning && tunePollTimer) {
      clearInterval(tunePollTimer);
      tunePollTimer = null;
    }
  }

  // Render overlaid training curves from tuning trial metrics
  async function renderTuneTrainCurves(stage) {
    const stageToFile = {
      A: "train_thinker.jsonl", B: "train_audio_enc.jsonl", C: "train_vision.jsonl",
      D: "train_talker.jsonl", E: "sft_omni.jsonl", F: "train_vocoder.jsonl", G: "train_ocr.jsonl",
    };
    const metricsFile = stageToFile[stage];
    if (!metricsFile) return;

    const res = await api.get(`/api/metrics/data?file=${metricsFile}`);
    if (!res.ok || !res.rows || !res.rows.length) return;

    // Filter rows from tuning runs (save_dir contains "tune_")
    const tuneRows = res.rows.filter(r =>
      r.phase !== "event" &&
      (r.metric_name === "loss" || r.metric_name === "val_loss") &&
      String(r.run_id || "").length > 0
    );
    if (!tuneRows.length) return;

    // Group by (run_id, metric_name)
    const groups = new Map();
    for (const r of tuneRows) {
      const key = `${r.run_id}|${r.metric_name}`;
      if (!groups.has(key)) groups.set(key, { metric: r.metric_name, run: r.run_id, points: [] });
      const x = Number(r.step ?? 0);
      const y = Number(r.metric_value ?? 0);
      if (Number.isFinite(x) && Number.isFinite(y)) groups.get(key).points.push([x, y]);
    }

    // Sort points and build series
    const palette = ["#8b5cf6", "#22d3ee", "#f59e0b", "#10b981", "#ef4444", "#f97316", "#60a5fa", "#a3e635", "#e879f7", "#fb923c"];
    const series = [];
    let idx = 0;
    const runs = [...new Set([...groups.values()].map(g => g.run))];

    for (const [key, g] of groups) {
      g.points.sort((a, b) => a[0] - b[0]);
      const runIdx = runs.indexOf(g.run);
      const color = palette[runIdx % palette.length];
      const isDashed = g.metric === "val_loss";
      series.push({
        name: `${g.metric} (${g.run.slice(0, 6)})`,
        type: "line",
        data: g.points,
        symbol: "none",
        lineStyle: { width: isDashed ? 1.5 : 1, color, type: isDashed ? "dashed" : "solid", opacity: 0.7 },
        large: true,
        sampling: "lttb",
      });
    }

    if (!series.length) return;

    if (!tuneTrainChart) tuneTrainChart = echarts.init($("#tuneTrainChart"));
    tuneTrainChart.setOption({
      animation: false,
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", backgroundColor: "rgba(15,18,38,0.9)", borderColor: "rgba(255,255,255,0.15)", textStyle: { color: "#e8ecf4", fontSize: 10 } },
      legend: { show: false },
      grid: { left: 60, right: 20, top: 20, bottom: 30 },
      xAxis: {
        type: "value", name: "Step", nameTextStyle: { color: "#9ca3bf" },
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)" } },
      },
      yAxis: {
        type: "value", name: "Loss", nameTextStyle: { color: "#9ca3bf" },
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
      },
      dataZoom: [{ type: "inside" }],
      series,
    }, true);
  }

  // Apply best config buttons
  $("#tuneApplyNewBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    const res = await api.post("/api/tuning/apply", { stage, target: "new" });
    const el = $("#tuneApplyResult");
    if (res.ok) {
      const diff = Object.entries(res.applied).map(([k, v]) =>
        `  ${k}: ${JSON.stringify(v.old)} → ${JSON.stringify(v.new)}`
      ).join("\n");
      el.innerHTML = `<span style="color:var(--success)">Saved to ${res.saved_to}</span> (best val_loss: ${res.best_val_loss?.toFixed(4)})\n<pre style="font-size:10px;margin-top:4px;color:var(--text-soft)">${diff}</pre>`;
      showToast(`Best config saved to ${res.saved_to}`, "success");
    } else {
      el.textContent = `Error: ${res.error}`;
    }
  });

  $("#tuneApplyBaseBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    if (!confirm(`Overwrite the base config for stage ${stage}? This will modify configs/${tuneSpaces[stage]?.config || "..."}.`)) return;
    const res = await api.post("/api/tuning/apply", { stage, target: "base" });
    const el = $("#tuneApplyResult");
    if (res.ok) {
      const diff = Object.entries(res.applied).map(([k, v]) =>
        `  ${k}: ${JSON.stringify(v.old)} → ${JSON.stringify(v.new)}`
      ).join("\n");
      el.innerHTML = `<span style="color:var(--warning)">Base config overwritten: ${res.saved_to}</span>\n<pre style="font-size:10px;margin-top:4px;color:var(--text-soft)">${diff}</pre>`;
      showToast(`Base config updated for stage ${stage}`, "warning");
    } else {
      el.textContent = `Error: ${res.error}`;
    }
  });

  // Clear tuning data
  $("#tuneClearBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    if (!confirm(`Delete ALL tuning data for stage ${stage}? (DB, trial checkpoints, tuned config)`)) return;
    const res = await api.post("/api/tuning/clear", { stage });
    if (res.ok) {
      showToast(`Cleared ${res.count} items for stage ${stage}`, "success");
      pollTuningProgress();
    } else {
      showToast(res.error || "Failed to clear", "error");
    }
  });

  window._pollTuningProgress = pollTuningProgress;
  stageSel.addEventListener("change", () => { renderParams(); pollTuningProgress(); });
  renderParams();
  pollTuningProgress();

  // Start tuning
  $("#tuneStartBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    const nTrials = parseInt($("#tuneTrials").value) || 30;
    const maxSteps = parseInt($("#tuneSteps").value) || 2000;
    progressDiv.innerHTML = `<span class="status-dot running"></span> Starting tuning for Stage ${stage}...`;
    const res = await api.post("/api/tuning/start", { stage, n_trials: nTrials, max_steps: maxSteps });
    if (res.ok) {
      progressDiv.innerHTML = `<span class="status-dot running"></span> Tuning started: Stage ${stage}, ${nTrials} trials (PID ${res.pid})`;
      // Start polling
      if (!tunePollTimer) tunePollTimer = setInterval(pollTuningProgress, 8000);
    } else {
      progressDiv.innerHTML = `<span style="color:var(--danger)">Error: ${res.error || "Failed to start"}</span>`;
    }
  });

  // Stop tuning
  $("#tuneStopBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    if (!confirm(`Stop tuning for stage ${stage}?`)) return;
    const res = await api.post("/api/tuning/stop", { stage });
    progressDiv.innerHTML = res.ok
      ? `<span style="color:var(--warning)">Stopped tuning for Stage ${stage}</span>`
      : `<span style="color:var(--danger)">Error: ${res.error}</span>`;
    if (tunePollTimer) { clearInterval(tunePollTimer); tunePollTimer = null; }
    pollTuningProgress();
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
