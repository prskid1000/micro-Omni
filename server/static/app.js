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
  gpuHistory: [],
  gpuSparkChart: null,
  logAutoRefreshTimer: null,
};

// ── Toast notifications ─────────────────────────────────────
function showToast(message, type = "info", duration = 5000) {
  const stack = $("#toastStack");
  if (!stack) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icon = type === "success" ? "&#10003;" : type === "error" ? "&#10007;" : type === "warning" ? "&#9888;" : "&#8505;";
  toast.innerHTML = `<span>${icon}</span> ${message}`;
  toast.addEventListener("click", () => toast.remove());
  stack.appendChild(toast);
  if (duration > 0) setTimeout(() => toast.remove(), duration);
  // Max 3 visible
  while (stack.children.length > 3) stack.children[0].remove();
}

// ── Debounce helper ─────────────────────────────────────────
function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

// ── Relative timestamp helper ───────────────────────────────
function timeAgo(iso) {
  if (!iso) return "";
  const sec = (Date.now() - new Date(iso).getTime()) / 1000;
  if (sec < 60) return Math.round(sec) + "s ago";
  if (sec < 3600) return Math.round(sec / 60) + "m ago";
  if (sec < 86400) return Math.round(sec / 3600) + "h ago";
  return Math.round(sec / 86400) + "d ago";
}

// ── Desktop notifications (#19) ─────────────────────────────
function _desktopNotify(body) {
  if (typeof Notification === "undefined") return;
  if (Notification.permission === "granted") {
    new Notification("micro-Omni", { body });
  } else if (Notification.permission !== "denied") {
    Notification.requestPermission();
  }
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

// ── ECharts registry & init ─────────────────────────────────
const chartRegistry = [];

function registerChart(instance) {
  chartRegistry.push(instance);
  return instance;
}

function initChart() {
  state.chart = registerChart(echarts.init($("#chart"), null, { renderer: "canvas" }));
  window.addEventListener("resize", () => {
    chartRegistry.forEach(c => { if (!c.isDisposed()) c.resize(); });
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

  // Event annotations (#15): find events and add markLines to first series
  const eventRows = state.allRows.filter(r => r.phase === "event");
  const xKey2 = xKey; // capture for event filtering
  const eventMarkLines = eventRows
    .filter(r => Number.isFinite(Number(r[xKey2] ?? 0)))
    .map(r => ({
      xAxis: Number(r[xKey2]),
      label: { formatter: r.metric_name || "event", fontSize: 9, color: "#f59e0b" },
      lineStyle: { color: "rgba(245,158,11,0.4)", type: "dashed", width: 1 },
    }));
  if (eventMarkLines.length > 0 && series.length > 0) {
    series[0].markLine = {
      silent: true,
      symbol: "none",
      data: eventMarkLines,
    };
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
    legend: { show: false },
    grid: { left: 60, right: state.dualAxis ? 70 : 40, top: 40, bottom: 60 },
    toolbox: {
      feature: {
        saveAsImage: { title: "Save" },
        dataZoom: { title: { zoom: "Zoom", back: "Reset" } },
        restore: { title: "Restore" },
      },
      iconStyle: { borderColor: "#9ca3bf" },
      right: 4,
      top: 0,
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

  // Store legend data for the series popover
  state._chartLegend = legendData;
}

// ── Series popover ──────────────────────────────────────────
function openSeriesPopover(btnEl) {
  const existing = document.querySelector(".series-popover");
  if (existing) { existing.remove(); return; }

  // Use the chart's actual current series (what's rendered, not all data)
  const legendData = state._chartLegend || [];
  if (!legendData.length || !state.chart) return;

  const pop = document.createElement("div");
  pop.className = "run-popover series-popover";

  // Get current visibility from chart
  const opt = state.chart.getOption();
  const legendSelected = (opt.legend && opt.legend[0] && opt.legend[0].selected) || {};

  let html = '<div class="run-popover-title">Chart Series</div>';
  html += '<div class="run-popover-list">';
  for (const name of legendData) {
    if (name.includes("(raw)")) continue; // skip raw lines
    const checked = legendSelected[name] !== false ? "checked" : "";
    const shortName = name.length > 50 ? name.slice(0, 47) + "..." : name;
    html += `<label class="run-popover-item"><input type="checkbox" value="${name}" ${checked} /> <span title="${name}">${shortName}</span></label>`;
  }
  html += '</div>';
  html += '<div class="run-popover-actions">';
  html += '<button class="btn btn-sm" data-action="all">All</button>';
  html += '<button class="btn btn-sm" data-action="none">None</button>';
  html += '<button class="btn btn-sm btn-accent" data-action="apply">Apply</button>';
  html += '</div>';
  pop.innerHTML = html;

  const rect = btnEl.getBoundingClientRect();
  pop.style.position = "fixed";
  pop.style.top = (rect.bottom + 4) + "px";
  pop.style.right = (window.innerWidth - rect.right) + "px";
  pop.style.left = "auto";
  document.body.appendChild(pop);

  pop.querySelector('[data-action="all"]').addEventListener("click", () => {
    pop.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
  });
  pop.querySelector('[data-action="none"]').addEventListener("click", () => {
    pop.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
  });

  pop.querySelector('[data-action="apply"]').addEventListener("click", () => {
    const selected = {};
    for (const name of legendData) {
      const cb = pop.querySelector(`input[value="${CSS.escape(name)}"]`);
      selected[name] = cb ? cb.checked : true;
      // Also handle raw line visibility
      const rawName = name + " (raw)";
      if (legendData.includes(rawName)) selected[rawName] = selected[name];
    }
    // Update chart legend selection (hidden legend still controls visibility)
    state.chart.setOption({ legend: { show: false, selected } });
    pop.remove();
  });

  setTimeout(() => {
    document.addEventListener("click", function closer(e) {
      if (!pop.contains(e.target) && e.target !== btnEl) {
        pop.remove();
        document.removeEventListener("click", closer);
      }
    });
  }, 10);
}

// ── Filtering ───────────────────────────────────────────────
function getSelected(sel) {
  return new Set([...sel.selectedOptions].map(o => o.value));
}

function getFilteredRows() {
  const files = getSelected(dom.fileSelect);
  const runs = getSelected(dom.runSelect);
  const metrics = getSelected(dom.metricSelect);

  // Expand __more_runs__ virtual group into actual tune run_ids
  let expandedRuns = runs;
  if (runs.has("__more_runs__") && state._tuneRunIds) {
    expandedRuns = new Set(runs);
    expandedRuns.delete("__more_runs__");
    for (const rid of state._tuneRunIds) expandedRuns.add(rid);
  }

  return state.allRows.filter(r => {
    if (r.phase === "event") return false;
    if (files.size && !files.has(r._file)) return false;
    if (expandedRuns.size && !expandedRuns.has(String(r.run_id || ""))) return false;
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

  // Preserve current selection, or restore from localStorage
  let prevActive = new Set([...sel.selectedOptions].map(o => o.value));
  if (prevActive.size === 0) {
    try {
      const saved = localStorage.getItem("filters_" + containerId);
      if (saved) prevActive = new Set(JSON.parse(saved).filter(v => values.includes(v)));
    } catch {}
  }

  // Build hidden select
  sel.innerHTML = values.map(v =>
    `<option value="${v}"${prevActive.has(v) ? " selected" : ""}>${v}</option>`
  ).join("");

  // Build chips with "x" button on active ones
  let html = "";
  for (const v of values) {
    const isActive = prevActive.has(v);
    // Use friendly label if available (e.g. "Train (b376af) s8400" for run IDs)
    const friendlyLabel = (state._runLabels && state._runLabels[v]) || null;
    const label = friendlyLabel || (v.length > 24 ? v.slice(0, 22) + ".." : v);
    const xBtn = isActive ? '<span class="chip-x">&times;</span>' : "";
    html += `<span class="chip${isActive ? " active" : ""}" data-value="${v}" title="${v}">${label}${xBtn}</span>`;
  }
  // Clear All button (only show if any active)
  if (prevActive.size > 0) {
    html += `<span class="chip chip-clear" data-action="clear-all">Clear All</span>`;
  }
  container.innerHTML = html;

  function syncAndUpdate() {
    const activeValues = new Set(
      [...container.querySelectorAll(".chip.active:not(.chip-clear)")].map(c => c.dataset.value)
    );
    [...sel.options].forEach(opt => { opt.selected = activeValues.has(opt.value); });
    try { localStorage.setItem("filters_" + containerId, JSON.stringify([...activeValues])); } catch {}
    updateChart();
    renderLatestValues();
  }

  // Click handlers
  container.querySelectorAll(".chip:not(.chip-clear)").forEach(chip => {
    chip.addEventListener("click", (e) => {
      const val = chip.dataset.value;

      // "+N more" chip — open popover to select individual runs
      if (val === "__more_runs__" && state._allRunInfos && !e.target.classList.contains("chip-x")) {
        openRunPopover(chip, container, sel, containerId, selectId, values);
        return;
      }

      // If clicking the X button, deselect
      if (e.target.classList.contains("chip-x")) {
        chip.classList.remove("active");
        const x = chip.querySelector(".chip-x");
        if (x) x.remove();
      } else {
        chip.classList.toggle("active");
        if (chip.classList.contains("active") && !chip.querySelector(".chip-x")) {
          chip.insertAdjacentHTML("beforeend", '<span class="chip-x">&times;</span>');
        } else {
          const x = chip.querySelector(".chip-x");
          if (x) x.remove();
        }
      }
      syncAndUpdate();
      buildChips(containerId, selectId, values);
    });
  });

  // Clear All button
  const clearBtn = container.querySelector(".chip-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      [...sel.options].forEach(opt => { opt.selected = false; });
      try { localStorage.removeItem("filters_" + containerId); } catch {}
      buildChips(containerId, selectId, values);
      updateChart();
      renderLatestValues();
    });
  }
}

// ── Run popover for "+N more" chip ──────────────────────────
function openRunPopover(chipEl, container, sel, containerId, selectId, values) {
  // Close existing popover
  const existing = document.querySelector(".run-popover");
  if (existing) { existing.remove(); return; }

  const allRuns = state._allRunInfos || [];
  const selectedRuns = getSelected(sel);

  const pop = document.createElement("div");
  pop.className = "run-popover";

  let html = '<div class="run-popover-title">Select Runs</div>';
  html += '<div class="run-popover-list">';
  for (const r of allRuns) {
    const checked = selectedRuns.has(r.rid) ? "checked" : "";
    const label = `${r.rid.slice(0, 8)} (step ${r.maxStep}, ${r.count} rows)`;
    html += `<label class="run-popover-item"><input type="checkbox" value="${r.rid}" ${checked} /> ${label}</label>`;
  }
  html += '</div>';
  html += '<div class="run-popover-actions">';
  html += '<button class="btn btn-sm" data-action="all">All</button>';
  html += '<button class="btn btn-sm" data-action="none">None</button>';
  html += '<button class="btn btn-sm btn-accent" data-action="apply">Apply</button>';
  html += '</div>';
  pop.innerHTML = html;

  // Position below the chip
  const rect = chipEl.getBoundingClientRect();
  pop.style.position = "fixed";
  pop.style.top = (rect.bottom + 4) + "px";
  pop.style.left = rect.left + "px";
  document.body.appendChild(pop);

  // Select All / None
  pop.querySelector('[data-action="all"]').addEventListener("click", () => {
    pop.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
  });
  pop.querySelector('[data-action="none"]').addEventListener("click", () => {
    pop.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
  });

  // Apply
  pop.querySelector('[data-action="apply"]').addEventListener("click", () => {
    const checked = new Set([...pop.querySelectorAll('input:checked')].map(cb => cb.value));
    // Update hidden select
    [...sel.options].forEach(opt => { opt.selected = checked.has(opt.value); });
    // Also add checked runs that aren't in the select as options
    for (const rid of checked) {
      if (![...sel.options].some(o => o.value === rid)) {
        const opt = document.createElement("option");
        opt.value = rid;
        opt.selected = true;
        sel.appendChild(opt);
      }
    }
    try { localStorage.setItem("filters_" + containerId, JSON.stringify([...checked])); } catch {}
    pop.remove();
    updateChart();
    renderLatestValues();
    // Rebuild chips to reflect new selection
    buildChips(containerId, selectId, values);
  });

  // Close on click outside
  setTimeout(() => {
    document.addEventListener("click", function closer(e) {
      if (!pop.contains(e.target) && e.target !== chipEl) {
        pop.remove();
        document.removeEventListener("click", closer);
      }
    });
  }, 10);
}

// Stage to metrics file mapping
const STAGE_METRICS_FILE = {
  A: "train_thinker.jsonl", B: "train_audio_enc.jsonl", C: "train_vision.jsonl",
  D: "train_talker.jsonl", E: "sft_omni.jsonl", F: "train_vocoder.jsonl", G: "train_ocr.jsonl",
};

function getActiveStage() {
  if (!state.pipelineData) return null;
  for (const [id, s] of Object.entries(state.pipelineData)) {
    if (s.status === "running") return id;
  }
  return null;
}

// Compute the run_id for a stage's real checkpoint (not tuning trials)
function getTrainingRunId(file) {
  // Find the run_id with the highest step in this file,
  // preferring runs with many rows (real training has 100s of rows, tuning trials have <20)
  const runCounts = {};
  const runMaxStep = {};
  for (const r of state.allRows) {
    if (r._file !== file || r.phase === "event") continue;
    const rid = r.run_id || "";
    runCounts[rid] = (runCounts[rid] || 0) + 1;
    const step = r.step || 0;
    if (step > (runMaxStep[rid] || 0)) runMaxStep[rid] = step;
  }
  // Pick the run with most rows (real training >> tuning trial)
  let best = null, bestCount = 0;
  for (const [rid, count] of Object.entries(runCounts)) {
    if (count > bestCount) { bestCount = count; best = rid; }
  }
  return best;
}

function getLatestRunId(file) {
  // Find the run_id with the highest step in this file
  let best = null, bestStep = -1;
  for (const r of state.allRows) {
    if (r._file !== file || r.phase === "event") continue;
    const step = r.step || 0;
    if (step > bestStep) { bestStep = step; best = r.run_id; }
  }
  return best;
}

function getLatestStepFromMetrics(file, runId, sinceTime) {
  let maxStep = 0;
  for (const r of state.allRows) {
    if (r._file !== file || r.phase === "event") continue;
    if (runId && r.run_id !== runId) continue;
    if (sinceTime && r.timestamp && r.timestamp < sinceTime) continue;
    const step = r.step || 0;
    if (step > maxStep) maxStep = step;
  }
  return maxStep;
}

function updateFilters() {
  const rows = state.allRows.filter(r => r.phase !== "event");
  const files = [...new Set(rows.map(r => r._file))].sort();
  const runs = [...new Set(rows.map(r => String(r.run_id || "")))].sort();
  const metrics = [...new Set(rows.map(r => String(r.metric_name || "")))].sort();

  // Auto-focus: if a stage is running and user hasn't manually selected filters,
  // pre-select the active stage's file and latest run_id
  const activeStage = getActiveStage();
  const activeFile = activeStage ? STAGE_METRICS_FILE[activeStage] : null;
  const userHasFilters = getSelected(dom.fileSelect).size > 0 || getSelected(dom.runSelect).size > 0;

  buildChips("fileChips", "fileSelect", files);
  // Classify and group runs: show top runs individually, group small ones
  const runInfos = runs.map(rid => {
    const runRows = rows.filter(r => String(r.run_id || "") === rid);
    const count = runRows.length;
    const maxStep = Math.max(0, ...runRows.map(r => r.step || 0));
    return { rid, count, maxStep };
  }).sort((a, b) => b.count - a.count);

  state._runLabels = {};
  state._tuneRunIds = null;
  const visibleRuns = [];

  // Only show Browse button — all run selection happens in popover
  state._allRunInfos = runInfos;
  state._tuneRunIds = null;

  if (runInfos.length > 0) {
    visibleRuns.push("__more_runs__");
    state._runLabels = { "__more_runs__": `Runs (${runInfos.length})` };
  }

  buildChips("runChips", "runSelect", visibleRuns);
  buildChips("metricChips", "metricSelect", metrics);

  // Auto-select active stage's file and run if user hasn't manually filtered
  if (activeFile && !userHasFilters && files.includes(activeFile)) {
    // Activate file chip
    const fileChip = document.querySelector(`#fileChips .chip[data-value="${activeFile}"]`);
    if (fileChip && !fileChip.classList.contains("active")) {
      fileChip.click();
    }
    // Activate latest run_id chip
    const latestRun = getLatestRunId(activeFile);
    if (latestRun) {
      const runChip = document.querySelector(`#runChips .chip[data-value="${latestRun}"]`);
      if (runChip && !runChip.classList.contains("active")) {
        runChip.click();
      }
    }
  }

  fillSelect(dom.fileSelect, files);
  fillSelect(dom.runSelect, visibleRuns);
  fillSelect(dom.metricSelect, metrics);
}

// ── Pipeline rendering ──────────────────────────────────────
function renderPipeline(stages) {
  if (!stages) { dom.pipeline.innerHTML = "<span style='color:var(--text-dim)'>Loading...</span>"; return; }

  dom.pipeline.innerHTML = Object.entries(stages).map(([id, s]) => {
    const statusClass = s.status;
    const meta = s.metadata;
    const proc = s.process;
    const maxSteps = meta?.max_steps || null;

    // Find current run_id and latest step from metrics (more accurate than metadata)
    const stageFile = STAGE_METRICS_FILE[id];
    const currentRunId = stageFile ? getLatestRunId(stageFile) : null;
    const runIdShort = currentRunId ? currentRunId.slice(0, 8) : "";
    // When running, only count metrics logged after process started (avoids stale pre-crash steps)
    const procStartTime = (s.status === "running" && proc && proc.start_time) ? proc.start_time : null;
    const metricsStep = stageFile ? getLatestStepFromMetrics(stageFile, currentRunId, procStartTime) : 0;
    const metaStep = meta ? (meta.step || 0) : 0;
    // When running: prefer metrics step (live progress), fall back to checkpoint step
    const liveStep = s.status === "running" ? (metricsStep || metaStep) : Math.max(metricsStep, metaStep);

    // Show single step number — no confusing dual display
    let stepInfo = "";
    const maxLabel = maxSteps ? `/${maxSteps}` : "";
    if (liveStep > 0) {
      stepInfo = `Step ${liveStep}${maxLabel}`;
    }

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
      case "paused":
        actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Resume</button>`;
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete and restart">Clear</button>`;
        break;
      case "failed":
        actions = `<button class="btn btn-sm" onclick="startStage('${id}')">Retry</button>`;
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete and restart">Clear</button>`;
        break;
      case "done":
        actions = `<button class="btn btn-sm" onclick="retrainStage('${id}')">Retrain</button>`;
        actions += ` <button class="btn btn-sm btn-danger" onclick="clearStage('${id}')" title="Delete checkpoint files">Clear</button>`;
        break;
      case "blocked":
        actions = `<span style="font-size:10px;color:var(--warning)">Needs: ${s.blocked_by.join(", ")}</span>`;
        break;
    }

    // Progress bar for running/paused stages
    let progressBar = "";
    const progressStep = liveStep || metaStep;
    if ((s.status === "running" || s.status === "paused") && maxSteps && progressStep > 0) {
      const pct = Math.min(100, Math.round((progressStep / maxSteps) * 100));
      progressBar = `<div class="stage-progress" style="width:${pct}%" title="${progressStep}/${maxSteps} (${pct}%)"></div>`;
    }

    return `
      <div class="stage-card">
        <div class="stage-id">Stage ${id}</div>
        <div class="stage-name">${s.name}</div>
        <div class="stage-status">
          <span class="status-dot ${statusClass}"></span>
          ${statusLine}
        </div>
        ${runIdShort ? `<div class="stage-run-id" title="Run ID: ${currentRunId}">run: ${runIdShort}</div>` : ""}
        <div class="stage-actions">${actions}</div>
        ${progressBar}
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

window.retrainStage = async function(stage) {
  if (!confirm(`Retrain stage ${stage} from scratch?`)) return;
  startStage(stage);
};

window.startAllIdle = async function() {
  if (!state.pipelineData) return;
  const idleStages = Object.entries(state.pipelineData)
    .filter(([_, s]) => s.status === "idle")
    .map(([id]) => id);
  if (!idleStages.length) {
    showToast("No idle stages to start", "warning");
    return;
  }
  if (!confirm(`Start ${idleStages.length} idle stage(s): ${idleStages.join(", ")}?`)) return;
  for (const stage of idleStages) {
    const res = await api.post("/api/training/start", { stage });
    if (res.ok) {
      showToast(`Stage ${stage} started (PID ${res.pid})`, "success");
    } else {
      showToast(`Stage ${stage}: ${res.error || "Failed"}`, "error");
      break; // GPU busy, stop trying
    }
    // Wait briefly for GPU lock
    await new Promise(r => setTimeout(r, 1000));
  }
  poll();
};

// ── Summary cards ───────────────────────────────────────────
function updateSummaryCards() {
  const allNonEvent = state.allRows.filter(r => r.phase !== "event");
  if (!allNonEvent.length) { resetSummaryCards(); return; }

  // Stage-aware filtering (#14): detect running stage and filter to its metrics file
  const stageToFile = {
    A: "train_thinker.jsonl", B: "train_audio_enc.jsonl", C: "train_vision.jsonl",
    D: "train_talker.jsonl", E: "sft_omni.jsonl", F: "train_vocoder.jsonl", G: "train_ocr.jsonl",
  };
  let activeFile = null;
  let isTuningActive = false;

  // Check if tuning is running (writes to same metrics file as training)
  try {
    const tuneProcs = state._tuneStatus || {};
    isTuningActive = Object.values(tuneProcs).some(p => p.status === "running");
  } catch {}

  if (state.pipelineData) {
    // Find running training stage
    for (const [id, s] of Object.entries(state.pipelineData)) {
      if (s.status === "running") { activeFile = stageToFile[id]; break; }
    }
    // If nothing running, find most recently active (done/paused/stopped)
    if (!activeFile) {
      for (const [id, s] of Object.entries(state.pipelineData)) {
        if (["done", "paused", "stopped"].includes(s.status)) { activeFile = stageToFile[id]; break; }
      }
    }
  }

  // Filter to active file
  let activeRunId = null;
  let filteredRows = activeFile
    ? allNonEvent.filter(r => r._file === activeFile)
    : allNonEvent;

  if (activeFile && filteredRows.length) {
    // When tuning is active, show the latest run (tuning trial) — it's the current work
    // When training is active or idle, prefer the training run (most rows)
    activeRunId = isTuningActive ? getLatestRunId(activeFile) : getTrainingRunId(activeFile);
    if (activeRunId) {
      filteredRows = filteredRows.filter(r => r.run_id === activeRunId);
    }
  }

  const rows = filteredRows;
  if (!rows.length) return;

  // Show which stage/run the cards are tracking
  const activeStage = getActiveStage();
  const prefix = isTuningActive ? "Tuning" : activeStage ? `Stage ${activeStage}` : "";
  const label = prefix
    ? `${prefix}${activeRunId ? " / " + activeRunId.slice(0, 8) : ""}`
    : activeRunId ? `run: ${activeRunId.slice(0, 8)}` : "";
  const stepLabel = $("#cardStep .card-label");
  if (stepLabel) stepLabel.textContent = label ? `Current Step (${label})` : "Current Step";

  // Find latest training metrics — use all phases to find the latest step
  const trainRows = rows.filter(r => r.phase === "train" && r.metric_name === "loss");
  const valRows = rows.filter(r => (r.phase === "val" || r.metric_name === "val_loss"));
  const allMetricRows = rows.filter(r => r.step != null);

  // Current step — from any metric row with highest step
  if (allMetricRows.length) {
    const latest = allMetricRows.reduce((a, b) => (a.step || 0) > (b.step || 0) ? a : b);
    setCard("cardStep", latest.step);
  } else if (trainRows.length) {
    const latest = trainRows.reduce((a, b) => (a.step || 0) > (b.step || 0) ? a : b);
    setCard("cardStep", latest.step);
  }

  // Best val loss
  if (valRows.length) {
    const best = valRows.reduce((a, b) => (a.metric_value || Infinity) < (b.metric_value || Infinity) ? a : b);
    setCard("cardBestVal", best.metric_value?.toFixed(4));
  }

  // LR — check all rows with lr field, not just train loss rows
  const withLR = rows.filter(r => r.lr != null);
  if (withLR.length) {
    const latest = withLR.reduce((a, b) => (a.step || 0) > (b.step || 0) ? a : b);
    setCard("cardLR", latest.lr?.toExponential(2));
  }

  // Throughput & ETA
  // For throughput, use all train loss rows from this file (across runs) for more data points
  const allFileTrainRows = activeFile
    ? allNonEvent.filter(r => r._file === activeFile && r.phase === "train" && r.metric_name === "loss")
    : trainRows;
  const throughputRows = allFileTrainRows.length >= 2 ? allFileTrainRows : trainRows;

  if (throughputRows.length >= 2) {
    const sorted = [...throughputRows].sort((a, b) => {
      // Sort by timestamp (more reliable than step which can reset across runs)
      return (a.timestamp || "").localeCompare(b.timestamp || "");
    });
    const recent = sorted.slice(-20);
    if (recent.length >= 2) {
      const dt = (new Date(recent[recent.length - 1].timestamp) - new Date(recent[0].timestamp)) / 1000;
      const dSteps = Math.abs((recent[recent.length - 1].step || 0) - (recent[0].step || 0));
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

  // Loss Trend (#13) — rolling delta of last 20 train loss values
  if (trainRows.length >= 2) {
    const sorted = [...trainRows].sort((a, b) => (a.step || 0) - (b.step || 0));
    const recent = sorted.slice(-20).map(r => Number(r.metric_value ?? 0));
    if (recent.length >= 2) {
      const delta = recent[recent.length - 1] - recent[0];
      const sign = delta < -0.001 ? "-" : delta > 0.001 ? "+" : "";
      const el = setCard("cardLossTrend", sign + Math.abs(delta).toFixed(4));
      if (el) {
        el.className = "card-value";
        if (delta < -0.001) el.style.color = "var(--success)";
        else if (delta > 0.001) el.style.color = "var(--danger)";
        else el.style.color = "var(--warning)";
      }
    }
  }

  // GPU sparkline (#21) — store readings in circular buffer
  if (state.gpuData && state.gpuData.gpu) {
    state.gpuHistory.push(state.gpuData.gpu.utilization_percent || 0);
    if (state.gpuHistory.length > 60) state.gpuHistory.shift();
    renderGPUSparkline();
  }
}

function findMaxSteps() {
  // When tuning is active, each trial runs for a limited number of steps
  // (default 500, set via --max_steps in tune.py)
  const isTuning = state._tuneStatus &&
    Object.values(state._tuneStatus).some(p => p.status === "running");
  if (isTuning) {
    // Read from the tuning HTML input if available
    const tuneStepsEl = $("#tuneSteps");
    if (tuneStepsEl) return parseInt(tuneStepsEl.value) || 500;
    return 500;
  }

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

function resetSummaryCards() {
  for (const id of ["cardStep", "cardBestVal", "cardLR", "cardETA", "cardThroughput", "cardGPUMem", "cardLossTrend"]) {
    setCard(id, "--");
  }
  const stepLabel = $("#cardStep .card-label");
  if (stepLabel) stepLabel.textContent = "Current Step";
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
  dom.gpuBadge.className = "gpu-badge" + (g.memory_percent > 95 ? " error" : g.memory_percent > 80 ? " warn" : "");
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

  // Build a map of previous values for color coding (#12)
  const prevValues = new Map();
  for (const r of rows) {
    const key = `${r._file}|${r.metric_name}`;
    const cur = prevValues.get(key);
    if (!cur) { prevValues.set(key, [r]); }
    else { cur.push(r); }
  }

  const tbody = $("#latestTable tbody");
  tbody.innerHTML = sorted.map(r => {
    const key = `${r._file}|${r.metric_name}`;
    const history = (prevValues.get(key) || []).sort((a, b) => (a.step || 0) - (b.step || 0));
    let valStyle = "";
    if (history.length >= 2) {
      const prev = history[history.length - 2];
      const cur = r;
      const pv = Number(prev.metric_value ?? 0);
      const cv = Number(cur.metric_value ?? 0);
      if (cv < pv) valStyle = ' style="color:var(--success)"'; // improving (lower)
      else if (cv > pv) valStyle = ' style="color:var(--warning)"'; // worsening
    }
    const ts = r.timestamp || "";
    const fullTs = ts ? ts.replace("T", " ").slice(0, 19) : "";
    return `
    <tr>
      <td style="color:var(--accent-2)">${r.metric_name || ""}</td>
      <td${valStyle}>${Number(r.metric_value ?? 0).toFixed(6)}</td>
      <td>${r.step ?? ""}</td>
      <td>${r.epoch ?? ""}</td>
      <td>${r.lr != null ? Number(r.lr).toExponential(2) : ""}</td>
      <td title="${fullTs}">${timeAgo(ts) || fullTs}</td>
    </tr>
  `;
  }).join("");
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

  // Load full configs for each checkpoint that has one (parallel)
  const ckptsWithConfig = state.checkpointData.filter(c => c.has_config);
  const configResponses = await Promise.all(
    ckptsWithConfig.map(c => api.get(`/api/system/config/${c.name}`))
  );
  const cards = ckptsWithConfig.map((ckpt, i) => {
    const res = configResponses[i];
    return { name: ckpt.name, config: res.ok ? res.config : (ckpt.config || {}) };
  });

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
  tbody.innerHTML = events.map(r => {
    const ts = r.timestamp || "";
    const fullTs = ts ? ts.replace("T", " ").slice(0, 19) : "";
    return `
    <tr>
      <td style="color:var(--warning)">${r.metric_name || ""}</td>
      <td>${r._file || ""}</td>
      <td>${r.run_id || ""}</td>
      <td>${r.step ?? ""}</td>
      <td>${r.epoch ?? ""}</td>
      <td title="${fullTs}">${timeAgo(ts) || fullTs}</td>
    </tr>
  `;
  }).join("");
}

// ── GPU Sparkline (#21) ─────────────────────────────────────
function renderGPUSparkline() {
  const el = $("#gpuSparkline");
  if (!el) return;
  if (!state.gpuSparkChart) {
    state.gpuSparkChart = registerChart(echarts.init(el));
  }
  state.gpuSparkChart.setOption({
    animation: false,
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    xAxis: { show: false, type: "category", data: state.gpuHistory.map((_, i) => i) },
    yAxis: { show: false, type: "value", min: 0, max: 100 },
    series: [{
      type: "line", data: state.gpuHistory, symbol: "none", smooth: true,
      lineStyle: { width: 1.5, color: "#8b5cf6" },
      areaStyle: { color: "rgba(139,92,246,0.2)" },
    }],
  }, true);
}

// ── Polling ─────────────────────────────────────────────────
async function poll() {
  try {
    // Parallel fetches
    const metricsUrl = state.lastTimestamp
      ? `/api/metrics/data?file=__all__&since=${encodeURIComponent(state.lastTimestamp)}`
      : "/api/metrics/data?file=__all__";
    const [metricsRes, gpuRes, pipelineRes, ckptRes, testStatusRes, tuneStatusRes, exportStatusRes] = await Promise.all([
      api.get(metricsUrl),
      api.get("/api/system/gpu"),
      api.get("/api/training/pipeline"),
      api.get("/api/system/checkpoints"),
      api.get("/api/testing/status"),
      api.get("/api/tuning/status"),
      api.get("/api/export/status"),
    ]);

    // Metrics — incremental: append new rows on subsequent polls
    if (metricsRes.ok && metricsRes.file_data) {
      let newRowCount = 0;
      for (const [fname, rows] of Object.entries(metricsRes.file_data)) {
        for (const r of rows) {
          r._file = fname;
          state.allRows.push(r);
          newRowCount++;
          if (r.timestamp && (!state.lastTimestamp || r.timestamp > state.lastTimestamp)) {
            state.lastTimestamp = r.timestamp;
          }
        }
      }
      // Only update UI if we got new data or this is the first poll
      if (newRowCount > 0 || !state.lastTimestamp) {
        updateFilters();
        updateChart();
        renderLatestValues();
        renderEvents();
      }
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
            _desktopNotify(`Stage ${id} ${s.status}!`);
          }
          // Training failed
          if (old.status === "running" && s.status === "failed") {
            showToast(`Stage ${id} (${s.name}) failed! Check logs.`, "error", 15000);
            document.title = `[Failed] micro-Omni Dashboard`;
            _desktopNotify(`Stage ${id} ${s.status}!`);
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

    // Store tuning status for summary cards to detect active tuning
    if (tuneStatusRes.ok && tuneStatusRes.processes) {
      state._tuneStatus = tuneStatusRes.processes;
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

    // Resize all ECharts instances (charts initialized while tab was hidden get 0 width)
    setTimeout(() => {
      chartRegistry.forEach(c => { if (!c.isDisposed()) c.resize(); });
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

  // Smoothing (debounced)
  dom.smoothSlider.addEventListener("input", debounce(() => {
    state.smoothing = parseFloat(dom.smoothSlider.value);
    dom.smoothValue.textContent = state.smoothing.toFixed(2);
    updateChart();
  }, 80));

  // Series popover
  $("#seriesBtn").addEventListener("click", (e) => openSeriesPopover(e.currentTarget));

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

  // Delete selected metrics data (files or run IDs)
  $("#deleteMetricsBtn").addEventListener("click", async () => {
    const selFiles = getSelected(dom.fileSelect);
    const selRuns = getSelected(dom.runSelect);

    if (selFiles.size === 0 && selRuns.size === 0) {
      showToast("Select files or run IDs to delete", "warning");
      return;
    }

    const parts = [];
    if (selFiles.size) parts.push(`${selFiles.size} file(s)`);
    if (selRuns.size) parts.push(`${selRuns.size} run(s)`);
    if (!confirm(`Delete metrics data for ${parts.join(" and ")}? This cannot be undone.`)) return;

    // Delete selected files
    for (const f of selFiles) {
      const res = await api.post("/api/metrics/delete", { file: f });
      if (res.ok) showToast(`Deleted ${f}`, "success");
      else showToast(`Failed: ${res.error}`, "error");
    }

    // Delete selected run IDs from all files
    if (selRuns.size > 0) {
      const allFiles = await api.get("/api/metrics/files");
      for (const f of (allFiles.files || [])) {
        for (const rid of selRuns) {
          await api.post("/api/metrics/delete-run", { file: f, run_id: rid });
        }
      }
      showToast(`Deleted ${selRuns.size} run(s) from all files`, "success");
    }

    // Clear state and refresh
    try { localStorage.removeItem("filters_fileChips"); localStorage.removeItem("filters_runChips"); } catch {}
    state.allRows = [];
    state.lastTimestamp = null;
    resetSummaryCards();
    if (state.chart) state.chart.clear();
    $("#latestTable tbody").innerHTML = "";
    $("#eventsTable tbody").innerHTML = "";
    poll();
  });

  // Delete ALL metrics data
  $("#deleteAllMetricsBtn").addEventListener("click", async () => {
    if (!confirm("Delete ALL metrics data? This removes every .jsonl file in logs/metrics/. Cannot be undone.")) return;
    const res = await api.post("/api/metrics/delete", { file: "__all__" });
    if (res.ok) {
      showToast(`Deleted ${res.count} metrics files`, "success");
      state.allRows = [];
      state.lastTimestamp = null;
      resetSummaryCards();
      if (state.chart) state.chart.clear();
      $("#latestTable tbody").innerHTML = "";
      $("#eventsTable tbody").innerHTML = "";
      try { localStorage.removeItem("filters_fileChips"); localStorage.removeItem("filters_runChips"); localStorage.removeItem("filters_metricChips"); } catch {}
      poll();
    } else {
      showToast(`Failed: ${res.error}`, "error");
    }
  });

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

  // Logs — with auto-scroll and auto-refresh (#7)
  dom.loadLogsBtn.addEventListener("click", async () => {
    const stage = dom.logStageSelect.value;
    const res = await api.get(`/api/training/logs/${stage}`);
    dom.logOutput.textContent = res.ok ? res.lines.join("\n") : "Failed to load logs";
    dom.logOutput.scrollTop = dom.logOutput.scrollHeight;
  });

  const autoRefreshCb = $("#logAutoRefresh");
  if (autoRefreshCb) {
    autoRefreshCb.addEventListener("change", () => {
      if (state.logAutoRefreshTimer) { clearInterval(state.logAutoRefreshTimer); state.logAutoRefreshTimer = null; }
      if (autoRefreshCb.checked) {
        state.logAutoRefreshTimer = setInterval(async () => {
          const stage = dom.logStageSelect.value;
          const res = await api.get(`/api/training/logs/${stage}`);
          if (res.ok) {
            dom.logOutput.textContent = res.lines.join("\n");
            dom.logOutput.scrollTop = dom.logOutput.scrollHeight;
          }
        }, 3000);
      }
    });
  }

  // ── Inference UI ───────────────────────────────────────
  const chatMessages = $("#inferChatMessages");
  const chatInput = $("#inferChatInput");
  const chatSendBtn = $("#inferChatSendBtn");
  let chatHistory = [];

  // Mode changes checkpoint list
  async function updateInferCkptList() {
    const mode = dom.inferMode.value;
    if (mode === "normal") {
      dom.inferCkpt.style.display = "";
      const res = await api.get("/api/system/checkpoints");
      if (res.ok) {
        dom.inferCkpt.innerHTML = res.checkpoints
          .filter(c => c.has_model)
          .map(c => `<option value="${c.path}">${c.name}</option>`)
          .join("");
      }
    } else {
      dom.inferCkpt.style.display = "none";
      dom.inferCkpt.innerHTML = '<option value="export">export/</option>';
    }
  }
  window._updateInferCkptList = updateInferCkptList;
  dom.inferMode.addEventListener("change", updateInferCkptList);
  updateInferCkptList();

  // Chat / Single mode toggle
  $$(".infer-mode-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      $$(".infer-mode-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.dataset.mode;
      $("#inferChatMode").style.display = mode === "chat" ? "flex" : "none";
      $("#inferSingleMode").style.display = mode === "single" ? "block" : "none";
    });
  });

  // Chat message rendering
  function addChatMessage(role, text, meta = "") {
    // Remove empty state
    const empty = chatMessages.querySelector(".infer-chat-empty");
    if (empty) empty.remove();

    const div = document.createElement("div");
    div.className = `infer-chat-msg ${role}`;

    // Check for attachments
    const imgPath = $("#inferChatImage").value;
    const audioPath = $("#inferChatAudio").value;
    let attachHtml = "";
    if (role === "user") {
      if (imgPath) attachHtml += `<div class="msg-attachment">&#128247; ${imgPath}</div>`;
      if (audioPath) attachHtml += `<div class="msg-attachment">&#127908; ${audioPath}</div>`;
    }

    div.innerHTML = attachHtml + text + (meta ? `<div class="msg-meta">${meta}</div>` : "");
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }

  // Send chat message
  async function sendChatMessage() {
    const text = chatInput.value.trim();
    const imgPath = $("#inferChatImage").value.trim();
    const audioPath = $("#inferChatAudio").value.trim();
    const useOCR = $("#inferOCR").checked;

    if (!text && !imgPath && !audioPath) return;

    // Add user message
    addChatMessage("user", text || (imgPath ? "[Image]" : "[Audio]"));
    chatInput.value = "";
    chatHistory.push({ role: "user", text, image: imgPath, audio: audioPath });

    // Clear attachments
    $("#inferChatImage").value = "";
    $("#inferChatAudio").value = "";
    $("#inferAttachments").style.display = "none";

    // Show thinking indicator
    const thinkDiv = addChatMessage("assistant thinking", "Thinking...");
    chatSendBtn.disabled = true;

    // Call API
    const mode = dom.inferMode.value;
    const source = dom.inferCkpt.value;
    let res;

    if (mode === "normal") {
      res = await api.post("/api/inference/chat", {
        text: text || undefined,
        ckpt_dir: source,
        image_path: imgPath || undefined,
        audio_in: audioPath || undefined,
        use_ocr: useOCR,
      });
    } else if (mode === "standalone") {
      res = await api.post("/api/inference/standalone", { text, model_dir: source, max_tokens: 64 });
    } else {
      res = await api.post("/api/inference/huggingface", {
        text, model_dir: source, image_path: imgPath || undefined,
        audio_path: audioPath || undefined, multimodal: !!(imgPath || audioPath), max_tokens: 32,
      });
    }

    // Replace thinking with response
    thinkDiv.remove();
    chatSendBtn.disabled = false;

    if (res.ok) {
      const parts = [];
      if (res.response) parts.push(res.response);
      if (res.image_response) parts.push(res.image_response);
      if (res.audio_response) parts.push(res.audio_response);
      const responseText = parts.join("\n") || "(empty response)";
      const meta = `${res.mode || mode} | ${res.elapsed_ms}ms`;
      addChatMessage("assistant", responseText, meta);
      chatHistory.push({ role: "assistant", text: responseText });
    } else {
      addChatMessage("assistant", `Error: ${res.error || "Unknown error"}`);
    }
  }

  chatSendBtn.addEventListener("click", sendChatMessage);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
  });

  // Attach image/audio buttons
  $("#inferAttachImgBtn").addEventListener("click", () => {
    const path = prompt("Image path:", "data/images/test.jpg");
    if (path) {
      $("#inferChatImage").value = path;
      $("#inferAttachments").style.display = "flex";
      $("#inferAttachText").textContent = "&#128247; " + path;
    }
  });

  $("#inferAttachAudioBtn").addEventListener("click", () => {
    const path = prompt("Audio path:", "data/audio/test.wav");
    if (path) {
      $("#inferChatAudio").value = path;
      $("#inferAttachments").style.display = "flex";
      $("#inferAttachText").textContent = "&#127908; " + path;
    }
  });

  // Clear chat
  $("#inferClearChat").addEventListener("click", () => {
    chatHistory = [];
    chatMessages.innerHTML = `
      <div class="infer-chat-empty">
        <div class="infer-chat-empty-icon">&#956;</div>
        <div>micro-Omni Inference</div>
        <div style="font-size:11px;color:var(--text-dim);margin-top:4px">Type a message below or attach an image/audio file</div>
      </div>`;
  });

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
    if (!testChart) testChart = registerChart(echarts.init($("#testChart")));
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

    const responses = await Promise.all(allScripts.map(s => api.get(`/api/testing/results/${s}`)));
    responses.forEach((res, i) => {
      const script = allScripts[i];
      if (res.ok && res.results && res.results.length) {
        const metrics = {};
        for (const r of res.results) {
          if (r.phase !== "event" && r.metric_name && r.metric_value != null) {
            metrics[r.metric_name] = r.metric_value;
          }
        }
        if (Object.keys(metrics).length) allResults[script] = metrics;
      }
    });

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

    if (!testCompareChartInstance) testCompareChartInstance = registerChart(echarts.init($("#testCompareChart")));
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

  // Single-mode inference (side-by-side input/output)
  const inferBtn = $("#inferBtn");
  const inferResult = $("#inferResult");
  if (inferBtn) {
    inferBtn.addEventListener("click", () => withDisable(inferBtn, async () => {
      const mode = dom.inferMode.value;
      const text = $("#inferText")?.value.trim() || "";
      const source = dom.inferCkpt.value;
      const imagePath = $("#inferImage")?.value.trim() || null;
      const audioIn = $("#inferAudioIn")?.value.trim() || null;
      const audioOut = $("#inferAudioOut")?.value.trim() || null;
      const useOCR = $("#inferOCR")?.checked || false;

      if (!text && !imagePath && !audioIn) {
        inferResult.textContent = "Provide text, image, or audio input.";
        return;
      }

      inferResult.textContent = `Running ${mode} inference...`;
      inferResult.style.color = "var(--accent-2)";
      inferBtn.textContent = "Running...";

      let res;
      if (mode === "normal") {
        res = await api.post("/api/inference/chat", {
          text, ckpt_dir: source, image_path: imagePath,
          audio_in: audioIn, audio_out: audioOut, use_ocr: useOCR,
        });
      } else if (mode === "standalone") {
        res = await api.post("/api/inference/standalone", { text, model_dir: source, max_tokens: 64 });
      } else {
        res = await api.post("/api/inference/huggingface", {
          text, model_dir: source, image_path: imagePath, audio_path: audioIn,
          multimodal: !!(imagePath || audioIn), max_tokens: 32,
        });
      }

      inferBtn.textContent = "Run Inference";
      inferResult.style.color = "";
      if (res.ok) {
        const parts = [];
        if (res.response) parts.push(res.response);
        if (res.image_response) parts.push(`[Image] ${res.image_response}`);
        if (res.audio_response) parts.push(`[Audio] ${res.audio_response}`);
        if (res.audio_out) parts.push(`Audio saved: ${res.audio_out}`);
        parts.push(`\n[${res.mode || mode} | ${res.elapsed_ms}ms]`);
        inferResult.textContent = parts.join("\n");
      } else {
        inferResult.textContent = `Error: ${res.error || "Unknown error"}`;
      }
    }));
  }

  // Unload model
  dom.inferUnloadBtn.addEventListener("click", () => withDisable(dom.inferUnloadBtn, async () => {
    dom.inferUnloadBtn.textContent = "...";
    const res = await api.post("/api/inference/unload", {});
    if (res.ok) showToast("Model unloaded, VRAM freed", "success");
    else showToast("Failed to unload", "error");
    dom.inferUnloadBtn.textContent = "Unload";
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

  // Keyboard shortcuts (#6)
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); dom.liveToggle.click(); }
    if (e.key === "r" || e.key === "R") { e.preventDefault(); poll(); }
    if (e.key === "l" || e.key === "L") { dom.logScaleBtn.click(); }
  });
}

// ── HP Tuning ───────────────────────────────────────────────
async function setupTuning() {
  const stageSel = $("#tuneStage");
  const paramsDiv = $("#tuneParams");
  const metricsDiv = $("#tuneMetrics");
  const resultsDiv = $("#tuneResults");
  const progressDiv = $("#tuneProgress");

  let tuneSpaces = {};
  let tunePollTimer = null;
  let tuneChart = null;
  let tuneTrainChart = null;
  const tuneSliceInstances = {};  // param name -> echarts instance
  let selectedMetrics = {};  // stage -> Set of selected metric keys

  // Load search spaces (includes metrics now)
  const spacesRes = await api.get("/api/tuning/spaces");
  tuneSpaces = spacesRes.ok ? spacesRes.spaces : {};

  // Initialize default metric selections per stage
  for (const [stageId, space] of Object.entries(tuneSpaces)) {
    if (space.metrics) {
      selectedMetrics[stageId] = new Set(
        space.metrics.filter(m => m.default).map(m => m.key)
      );
    }
  }

  // Render optimization metric chips
  function renderMetrics() {
    const stage = stageSel.value;
    const space = tuneSpaces[stage];
    if (!space || !space.metrics) { metricsDiv.innerHTML = ""; return; }

    // Restore from saved config if available
    metricsDiv.innerHTML = space.metrics.map(m => {
      const sel = selectedMetrics[stage]?.has(m.key) ? " selected" : "";
      const arrow = m.direction === "minimize" ? "↓" : "↑";
      return `<div class="tune-metric-chip${sel}" data-key="${m.key}" data-direction="${m.direction}">
        <span class="direction ${m.direction}">${arrow}</span>
        <span class="metric-name">${m.name}</span>
      </div>`;
    }).join("");

    // Click handlers for chips
    metricsDiv.querySelectorAll(".tune-metric-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const key = chip.dataset.key;
        if (!selectedMetrics[stage]) selectedMetrics[stage] = new Set();
        if (selectedMetrics[stage].has(key)) {
          selectedMetrics[stage].delete(key);
          chip.classList.remove("selected");
        } else {
          selectedMetrics[stage].add(key);
          chip.classList.add("selected");
        }
      });
    });
  }

  // Get selected metrics for current stage
  function getSelectedMetricKeys() {
    const stage = stageSel.value;
    const sel = selectedMetrics[stage];
    return sel ? [...sel] : [];
  }

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

    // Update button label based on whether DB has existing trials
    const startBtn = $("#tuneStartBtn");

    // Load results from DB (updates as trials complete)
    const resData = await api.get(`/api/tuning/results/${stage}`);

    // Restore tuning config (stage/trials/steps/metrics) from saved file
    if (resData.ok && resData.tune_config) {
      const tc = resData.tune_config;
      if (tc.n_trials) $("#tuneTrials").value = tc.n_trials;
      if (tc.max_steps) $("#tuneSteps").value = tc.max_steps;
      // Restore metric selections from saved config
      if (tc.metrics && Array.isArray(tc.metrics)) {
        selectedMetrics[stage] = new Set(tc.metrics);
        renderMetrics();
      }
    }

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
          ? `<span style="color:var(--text-dim)">Paused: ${r.n_trials} trials completed</span>`
          : "";
      }

      // Update Start button label and disabled state based on running
      if (startBtn) {
        if (isRunning) {
          startBtn.textContent = "Running...";
          startBtn.disabled = true;
        } else {
          startBtn.textContent = r.n_trials > 0 ? "Resume Tuning" : "Start Tuning";
          startBtn.disabled = false;
        }
      }

      // Best result — show selected metrics info
      if (r.best_trial) {
        const metricKeys = getSelectedMetricKeys();
        const objectiveLabel = metricKeys.length > 0 && metricKeys.some(m => m !== "val_loss")
          ? `objective (${metricKeys.join(" + ")})`
          : "val_loss";
        const bestParams = Object.entries(r.best_trial.params || {})
          .map(([k, v]) => `<span class="hp-key">${k}</span>=<span class="hp-val">${typeof v === "number" ? v.toPrecision(4) : v}</span>`)
          .join(", ");
        resultsDiv.innerHTML = `
          <div style="margin-bottom:6px"><strong style="color:var(--success)">Best: ${objectiveLabel} = ${r.best_trial.value?.toFixed(4)}</strong> (trial #${r.best_trial.number})</div>
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
      if (!tuneChart) tuneChart = registerChart(echarts.init($("#tuneChart")));
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
          legend: { data: ["Objective", "Best So Far"], textStyle: { color: "#9ca3bf", fontSize: 11 }, top: 4 },
          grid: { left: 60, right: 20, top: 35, bottom: 30 },
          xAxis: {
            type: "value", name: "Trial", nameTextStyle: { color: "#9ca3bf" },
            axisLabel: { color: "#9ca3bf", fontSize: 10 },
            axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
          },
          yAxis: {
            type: "value", name: "Objective", nameTextStyle: { color: "#9ca3bf" },
            axisLabel: { color: "#9ca3bf", fontSize: 10 },
            splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
          },
          series: [
            {
              name: "Objective", type: "scatter", symbolSize: 8,
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
      // Filter out penalty trials (val_loss >= 50 is almost certainly a penalty)
      const realTrials = chartTrials.filter(t => t.value != null && t.value < 50);

      if (realTrials.length >= 2) {
        const paramNames = new Set();
        for (const t of realTrials) { for (const k of Object.keys(t.params || {})) paramNames.add(k); }
        const sortedParams = [...paramNames].sort();

        if (sliceContainer.children.length !== sortedParams.length) {
          sliceContainer.innerHTML = sortedParams.map(p =>
            `<div class="tune-slice-chart" id="slice-${p.replace(/[^a-zA-Z0-9_]/g, '_')}"></div>`
          ).join("");
          for (const k of Object.keys(tuneSliceInstances)) {
            try { tuneSliceInstances[k].dispose(); } catch {}
            delete tuneSliceInstances[k];
          }
        }

        // Compute val_loss range for color mapping
        const allVals = realTrials.map(t => t.value).filter(v => v != null);
        const minVal = Math.min(...allVals);
        const maxVal = Math.max(...allVals);

        for (const paramName of sortedParams) {
          const elId = `slice-${paramName.replace(/[^a-zA-Z0-9_]/g, '_')}`;
          const el = $(`#${elId}`);
          if (!el) continue;

          if (!tuneSliceInstances[paramName]) {
            tuneSliceInstances[paramName] = registerChart(echarts.init(el));
          }
          const chart = tuneSliceInstances[paramName];

          // Build points — handle booleans and None values
          const points = [];
          for (const t of realTrials) {
            if (t.params == null || t.params[paramName] === undefined) continue;
            let x = t.params[paramName];
            if (x === true) x = 1;
            else if (x === false) x = 0;
            else if (x === null || x === "null" || x === "None") x = 0;
            else if (typeof x === "string") {
              const parsed = parseFloat(x);
              x = isNaN(parsed) ? 0 : parsed;
            }
            if (Number.isFinite(x) && Number.isFinite(t.value)) {
              points.push([x, t.value]);
            }
          }

          if (points.length === 0) continue;

          // Pre-compute colors by rank
          const sortedByVal = [...points].sort((a, b) => a[1] - b[1]);
          const coloredData = points.map(p => {
            const rank = sortedByVal.findIndex(s => s[0] === p[0] && s[1] === p[1]) / Math.max(sortedByVal.length - 1, 1);
            const color = rank < 0.33 ? "#10b981" : rank < 0.66 ? "#f59e0b" : "#ef4444";
            return { value: p, itemStyle: { color } };
          });

          // Detect if param is boolean-like (only 0 and 1)
          const uniqueX = new Set(points.map(p => p[0]));
          const isBoolLike = uniqueX.size <= 2 && [...uniqueX].every(v => v === 0 || v === 1);

          chart.setOption({
            animation: false,
            backgroundColor: "transparent",
            title: { text: paramName, textStyle: { color: "#9ca3bf", fontSize: 11, fontWeight: 600 }, left: 8, top: 4 },
            tooltip: {
              trigger: "item",
              backgroundColor: "rgba(15,18,38,0.9)",
              borderColor: "rgba(255,255,255,0.15)",
              textStyle: { color: "#e8ecf4", fontSize: 10 },
            },
            grid: { left: 50, right: 12, top: 30, bottom: 28 },
            xAxis: {
              type: isBoolLike ? "category" : "value",
              data: isBoolLike ? ["false", "true"] : undefined,
              axisLabel: { color: "#7a82a0", fontSize: 9 },
              axisLine: { lineStyle: { color: "rgba(255,255,255,0.1)" } },
              splitLine: { show: false },
            },
            yAxis: {
              type: "value", name: "objective",
              nameTextStyle: { color: "#7a82a0", fontSize: 9 },
              axisLabel: { color: "#7a82a0", fontSize: 9 },
              splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)" } },
            },
            series: [{
              type: "scatter",
              symbolSize: 10,
              data: isBoolLike
                ? coloredData.map(d => ({ value: [d.value[0] === 1 ? "true" : "false", d.value[1]], itemStyle: d.itemStyle }))
                : coloredData,
            }],
          }, true);
        }
      } else if (chartTrials.length >= 3 && realTrials.length < 2) {
        sliceContainer.innerHTML = '<span style="color:var(--warning);font-size:11px">All trials have penalty val_loss (100.0). Clear data and re-run tuning — the val_loss reader has been fixed.</span>';
      } else {
        sliceContainer.innerHTML = '<span style="color:var(--text-dim);font-size:11px">Need 2+ completed trials with real val_loss for slice plots</span>';
      }

      // ── Training curves from tuning trial metrics ─────────
      await renderTuneTrainCurves(stageSel.value);

      // Render metric breakdown (sub-tab)
      renderMetricBreakdown(r.trials, stage);

    } else {
      progressDiv.innerHTML = isRunning
        ? '<span class="status-dot running"></span> Tuning starting...'
        : "";
      resultsDiv.textContent = "No tuning DB found for this stage. Start a tuning run.";
      $("#tuneTrialsTable tbody").innerHTML = "";
      $("#tuneSliceCharts").innerHTML = "";
      $("#tuneApplyResult").innerHTML = "";
      if (startBtn) { startBtn.textContent = "Start Tuning"; startBtn.disabled = false; }

      // Clear charts
      if (tuneChart) { tuneChart.clear(); }
      if (tuneTrainChart) { tuneTrainChart.clear(); }
      for (const k of Object.keys(tuneSliceInstances)) {
        try { tuneSliceInstances[k].dispose(); } catch {}
        delete tuneSliceInstances[k];
      }
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

    if (!tuneTrainChart) tuneTrainChart = registerChart(echarts.init($("#tuneTrainChart")));
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
      el.innerHTML = `<span style="color:var(--success)">Saved to ${res.saved_to}</span> (best objective: ${res.best_val_loss?.toFixed(4)})\n<pre style="font-size:10px;margin-top:4px;color:var(--text-soft)">${diff}</pre>`;
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
    if (!confirm(`Delete ALL tuning data for stage ${stage}?\n\nThis removes:\n- Optuna DB (all trial results)\n- Trial checkpoints\n- Tuned config file`)) return;
    const res = await api.post("/api/tuning/clear", { stage });
    if (res.ok) {
      showToast(`Cleared ${res.count} items for stage ${stage}`, "success");
      // Force clear all UI immediately
      resultsDiv.textContent = "Cleared. No tuning data.";
      progressDiv.innerHTML = "";
      $("#tuneTrialsTable tbody").innerHTML = "";
      $("#tuneSliceCharts").innerHTML = "";
      $("#tuneApplyResult").innerHTML = "";
      if (tuneChart) tuneChart.clear();
      if (tuneTrainChart) tuneTrainChart.clear();
      for (const k of Object.keys(tuneSliceInstances)) {
        try { tuneSliceInstances[k].dispose(); } catch {}
        delete tuneSliceInstances[k];
      }
      // Re-poll after short delay to confirm empty state
      setTimeout(pollTuningProgress, 500);
    } else {
      showToast(res.error || "Failed to clear", "error");
    }
  });

  // ── Sub-tab switching ────────────────────────────────────
  document.querySelectorAll(".tune-subtab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tune-subtab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tune-subtab-content").forEach(c => { c.classList.remove("active"); c.style.display = "none"; });
      tab.classList.add("active");
      const target = tab.dataset.subtab;
      const content = document.querySelector(`.tune-subtab-content[data-subtab="${target}"]`);
      if (content) { content.classList.add("active"); content.style.display = ""; }
      // Resize charts when tab becomes visible
      if (target === "overview") {
        if (tuneChart) tuneChart.resize();
        if (tuneTrainChart) tuneTrainChart.resize();
      } else if (target === "params") {
        for (const inst of Object.values(tuneSliceInstances)) { try { inst.resize(); } catch {} }
      } else if (target === "metrics") {
        if (tuneMetricBreakdownChart) tuneMetricBreakdownChart.resize();
      }
    });
  });

  // ── Metric Breakdown Chart ─────────────────────────────────
  let tuneMetricBreakdownChart = null;

  function renderMetricBreakdown(trials, stage) {
    const space = tuneSpaces[stage];
    if (!space || !space.metrics) return;
    const metricDefs = space.metrics;

    // For now we use the trial objective values — when test metrics are stored
    // in trial user_attrs (future), we'll read those. For now, show objective trend.
    const completedTrials = (trials || [])
      .filter(t => t.state === "COMPLETE" && t.value != null)
      .sort((a, b) => a.number - b.number);

    if (completedTrials.length < 2) {
      $("#tuneMetricBreakdownChart").innerHTML = '<span style="color:var(--text-dim);font-size:11px;padding:20px;display:block">Need 2+ completed trials for metric breakdown</span>';
      return;
    }

    // Build chart with objective trend + running best
    if (!tuneMetricBreakdownChart) {
      tuneMetricBreakdownChart = registerChart(echarts.init($("#tuneMetricBreakdownChart")));
    }

    let runningBest = [];
    let best = Infinity;
    for (const t of completedTrials) { best = Math.min(best, t.value); runningBest.push(best); }

    // Also compute improvement percentage from first to best
    const firstVal = completedTrials[0].value;
    const bestVal = Math.min(...completedTrials.map(t => t.value));
    const improvement = firstVal > 0 ? ((firstVal - bestVal) / firstVal * 100).toFixed(1) : "N/A";

    tuneMetricBreakdownChart.setOption({
      animation: false,
      backgroundColor: "transparent",
      title: {
        text: `Objective Trend (${improvement}% improvement)`,
        textStyle: { color: "#9ca3bf", fontSize: 12 },
        left: 8, top: 4,
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(15,18,38,0.9)",
        borderColor: "rgba(255,255,255,0.15)",
        textStyle: { color: "#e8ecf4", fontSize: 11 },
        formatter: params => {
          const lines = params.map(p => `${p.marker} ${p.seriesName}: ${p.value[1]?.toFixed(4)}`);
          return `Trial #${params[0].value[0]}<br/>` + lines.join("<br/>");
        },
      },
      legend: { data: ["Objective", "Running Best", "Median"], textStyle: { color: "#9ca3bf", fontSize: 10 }, top: 4, right: 10 },
      grid: { left: 60, right: 20, top: 40, bottom: 30 },
      xAxis: {
        type: "value", name: "Trial", nameTextStyle: { color: "#9ca3bf" },
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
      },
      yAxis: {
        type: "value", name: "Value", nameTextStyle: { color: "#9ca3bf" },
        axisLabel: { color: "#9ca3bf", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(255,255,255,0.06)" } },
      },
      series: [
        {
          name: "Objective", type: "scatter", symbolSize: 6,
          data: completedTrials.map(t => [t.number, t.value]),
          itemStyle: { color: "#8b5cf6" },
        },
        {
          name: "Running Best", type: "line", smooth: true, symbol: "none",
          data: completedTrials.map((t, i) => [t.number, runningBest[i]]),
          lineStyle: { color: "#10b981", width: 2 },
          areaStyle: { color: "rgba(16, 185, 129, 0.06)" },
        },
        {
          name: "Median", type: "line", symbol: "none",
          data: (() => {
            const vals = [];
            return completedTrials.map((t, i) => {
              vals.push(t.value);
              const sorted = [...vals].sort((a, b) => a - b);
              const mid = sorted[Math.floor(sorted.length / 2)];
              return [t.number, mid];
            });
          })(),
          lineStyle: { color: "#f59e0b", width: 1, type: "dashed" },
        },
      ],
    }, true);

    // Build metric values table with selected metrics info
    const sel = selectedMetrics[stage] || new Set();
    const selMetrics = metricDefs.filter(m => sel.has(m.key));
    const headerCols = selMetrics.map(m => {
      const arrow = m.direction === "minimize" ? "↓" : "↑";
      return `<th style="font-size:10px">${m.name} ${arrow}</th>`;
    }).join("");

    // Update table headers dynamically
    const thead = document.querySelector("#tuneMetricTable thead tr");
    if (thead) {
      thead.innerHTML = `<th>#</th><th>Objective</th>${headerCols}<th>State</th>`;
    }

    // Table body — show sorted by objective
    const sortedTrials = [...completedTrials].sort((a, b) => a.value - b.value);
    const tbody = document.querySelector("#tuneMetricTable tbody");
    if (tbody) {
      tbody.innerHTML = sortedTrials.slice(0, 50).map((t, i) => {
        const isBest = i === 0;
        // For now metric columns show "--" until we have per-trial test metric storage
        const metricCols = selMetrics.map(() => `<td style="font-size:10px;color:var(--text-dim)">--</td>`).join("");
        return `<tr${isBest ? ' style="background:rgba(16,185,129,0.08)"' : ""}>
          <td>${isBest ? "★ " : ""}${t.number}</td>
          <td>${t.value?.toFixed(4) ?? "--"}</td>
          ${metricCols}
          <td style="color:var(--success)">${t.state}</td>
        </tr>`;
      }).join("");
    }
  }

  window._pollTuningProgress = pollTuningProgress;
  stageSel.addEventListener("change", () => { renderMetrics(); renderParams(); pollTuningProgress(); });
  renderMetrics();
  renderParams();
  pollTuningProgress();

  // Start tuning
  $("#tuneStartBtn").addEventListener("click", async () => {
    const stage = stageSel.value;
    const nTrials = parseInt($("#tuneTrials").value) || 30;
    const maxSteps = parseInt($("#tuneSteps").value) || 2000;

    // Check if DB exists (resume vs fresh)
    const existing = await api.get(`/api/tuning/results/${stage}`);
    const existingTrials = (existing.results && !existing.results.error) ? existing.results.n_trials : 0;
    const isResume = existingTrials > 0;
    const label = isResume
      ? `Resuming from ${existingTrials} existing trials, adding ${nTrials} more...`
      : `Starting fresh: ${nTrials} trials...`;

    progressDiv.innerHTML = `<span class="status-dot running"></span> ${label}`;
    const metrics = getSelectedMetricKeys();
    const res = await api.post("/api/tuning/start", { stage, n_trials: nTrials, max_steps: maxSteps, metrics: metrics.length > 0 ? metrics : undefined });
    if (res.ok) {
      progressDiv.innerHTML = `<span class="status-dot running"></span> Tuning running: Stage ${stage}, ${nTrials} new trials (PID ${res.pid})${isResume ? " [resumed]" : ""}`;
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

// ── Config Diff (#16) ───────────────────────────────────────
function highlightConfigDiff(editorEl, ckptEl, baseConfig, ckptConfig) {
  const diffEl = $("#configDiffOutput");
  if (!diffEl || !baseConfig || !ckptConfig) { clearConfigDiff(); return; }

  const allKeys = new Set([...Object.keys(baseConfig), ...Object.keys(ckptConfig)]);
  const lines = [];
  for (const key of [...allKeys].sort()) {
    const bv = baseConfig[key];
    const cv = ckptConfig[key];
    const bs = JSON.stringify(bv);
    const cs = JSON.stringify(cv);
    if (bs === cs) {
      lines.push(`<span style="color:var(--success)">${key}: ${bs}</span>`);
    } else if (bv === undefined) {
      lines.push(`<span style="color:var(--info)">+ ${key}: ${cs} (checkpoint only)</span>`);
    } else if (cv === undefined) {
      lines.push(`<span style="color:var(--warning)">- ${key}: ${bs} (base only)</span>`);
    } else {
      lines.push(`<span style="color:var(--danger)">${key}: ${bs} -> ${cs}</span>`);
    }
  }
  diffEl.innerHTML = lines.join("\n");
}

function clearConfigDiff() {
  const diffEl = $("#configDiffOutput");
  if (diffEl) diffEl.innerHTML = "";
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
        // Config diff highlighting (#16)
        highlightConfigDiff(editor, ckptView, r.ok ? r.config : null, cr.config);
      } else {
        ckptView.value = "(not found)";
      }
    } else {
      ckptView.value = "";
      clearConfigDiff();
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
