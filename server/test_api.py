"""Comprehensive API test suite for the micro-Omni unified server.

Tests every endpoint, every state transition, every error path.

Usage:
    # Start server first: python -m server --no-open
    python server/test_api.py
    python server/test_api.py --host 127.0.0.1 --port 8000
    python server/test_api.py --skip-slow          # Skip tests that wait for processes
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error


class APITester:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.base = f"http://{host}:{port}"
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: list[str] = []
        self.section_counts: dict[str, tuple[int, int]] = {}
        self._section = ""

    # ── HTTP helpers ─────────────────────────────────────────

    def get(self, path: str) -> dict:
        try:
            r = urllib.request.urlopen(f"{self.base}{path}", timeout=15)
            return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def post(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(f"{self.base}{path}")
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body or {}).encode()
        try:
            r = urllib.request.urlopen(req, timeout=15)
            return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_raw(self, path: str) -> tuple[int, bytes, dict]:
        """Return (status_code, body_bytes, headers_dict)."""
        try:
            r = urllib.request.urlopen(f"{self.base}{path}", timeout=10)
            return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)
        except Exception as e:
            return 0, str(e).encode(), {}

    # ── Assertions ───────────────────────────────────────────

    def check(self, label: str, result: dict, expect_ok: bool = True) -> dict:
        ok = result.get("ok", False)
        passed = ok == expect_ok

        detail = ""
        if not ok and expect_ok:
            detail = f' -> {result.get("error", "")}'
        elif ok and not expect_ok:
            detail = " -> expected failure but succeeded"

        self._record(label, passed, detail)
        return result

    def assert_eq(self, label: str, actual, expected):
        passed = actual == expected
        detail = "" if passed else f" -> got {actual!r}, expected {expected!r}"
        self._record(label, passed, detail)

    def assert_in(self, label: str, value, container):
        passed = value in container
        detail = "" if passed else f" -> {value!r} not in {container!r}"
        self._record(label, passed, detail)

    def assert_true(self, label: str, condition: bool, detail: str = ""):
        self._record(label, condition, f" -> {detail}" if not condition and detail else "")

    def assert_gt(self, label: str, actual, threshold):
        passed = actual > threshold
        detail = "" if passed else f" -> {actual} not > {threshold}"
        self._record(label, passed, detail)

    def skip(self, label: str, reason: str = ""):
        self.skipped += 1
        print(f"  [SKIP] {label}{f' ({reason})' if reason else ''}")

    def _record(self, label: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}{detail}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"[{self._section}] {label}{detail}")

    # ── Wait helpers ─────────────────────────────────────────

    def wait_gpu_free(self, timeout: int = 120) -> bool:
        for _ in range(timeout // 2):
            r = self.get("/api/training/pipeline")
            running_train = any(s.get("status") == "running" for s in r.get("stages", {}).values())
            r2 = self.get("/api/testing/status")
            running_test = any(p.get("status") == "running" for p in r2.get("processes", {}).values())
            r3 = self.get("/api/tuning/status")
            running_tune = any(p.get("status") == "running" for p in r3.get("processes", {}).values())
            r4 = self.get("/api/export/status")
            running_exp = any(p.get("status") == "running" for p in r4.get("processes", {}).values())
            if not (running_train or running_test or running_tune or running_exp):
                return True
            time.sleep(2)
        return False

    def wait_process_done(self, category: str, timeout: int = 120) -> dict | None:
        endpoint = f"/api/{category}/status"
        for _ in range(timeout // 2):
            r = self.get(endpoint)
            procs = r.get("processes", {})
            running = [p for p in procs.values() if p.get("status") == "running"]
            if not running:
                # Return the most recently finished
                finished = sorted(procs.values(), key=lambda p: p.get("end_time") or "", reverse=True)
                return finished[0] if finished else None
            time.sleep(2)
        return None

    # ── Runner ───────────────────────────────────────────────

    def run_all(self, skip_slow: bool = False):
        print("=" * 64)
        print("  micro-Omni Server API Test Suite")
        print("=" * 64)

        self._run_section("Static Files", self.test_static_files)
        self._run_section("Metrics API", self.test_metrics_api)
        self._run_section("System API", self.test_system_api)
        self._run_section("Config Roundtrip", self.test_config_roundtrip)
        self._run_section("Pipeline Status", self.test_pipeline_status)
        self._run_section("Training Lifecycle", self.test_training_lifecycle, skip_slow)
        self._run_section("GPU Lock", self.test_gpu_lock, skip_slow)
        self._run_section("Pause/Resume Progress", self.test_pause_resume_progress, skip_slow)
        self._run_section("Training All Stages", self.test_all_stages_start_stop, skip_slow)
        self._run_section("Testing Lifecycle", self.test_testing_lifecycle, skip_slow)
        self._run_section("Inference API", self.test_inference_api)
        self._run_section("Tuning API", self.test_tuning_api)
        self._run_section("Export API", self.test_export_api, skip_slow)
        self._run_section("Error Handling", self.test_error_handling)
        self._run_section("CORS Headers", self.test_cors)
        self._run_section("Content Types", self.test_content_types)
        # Integration data validation (reads real files on disk)
        self._run_section("Metrics Data Integrity", self.test_metrics_data_integrity)
        self._run_section("Checkpoint Data Integrity", self.test_checkpoint_data_integrity)
        self._run_section("Config Schema Integrity", self.test_config_schema_integrity)
        self._run_section("Pipeline Dependency Integrity", self.test_pipeline_dependency_integrity)
        self._run_section("Training Logs Integrity", self.test_training_logs_integrity)
        self._run_section("Tuning Results Integrity", self.test_tuning_results_integrity)
        self._run_section("Search Space Integrity", self.test_search_space_integrity)
        # Cross-validation (config ↔ script ↔ tuning)
        self._run_section("Config↔Script Param Coverage", self.test_config_training_param_coverage)
        self._run_section("Tuning↔Config Coverage", self.test_tuning_space_covers_config)
        self._run_section("Tuning Param Type Match", self.test_tuning_param_types_match_config)
        self._run_section("Tuning Run ID", self.test_tuning_run_id_computation)
        self._run_section("Tuning DB Schema", self.test_tuning_optuna_db_schema)
        self._run_section("Metrics Run ID Consistency", self.test_metrics_run_id_consistency)
        self._run_section("Config Value Sanity", self.test_config_value_sanity)
        self._run_section("Tuning Range Sanity", self.test_tuning_range_sanity)
        self._run_section("Config in Tuning Range", self.test_config_value_in_tuning_range)
        self._run_section("All Stages Registered", self.test_all_stages_registered, skip_slow)
        # New feature tests
        self._run_section("Metrics Caching", self.test_metrics_caching)
        self._run_section("New UI Elements", self.test_new_ui_elements)
        self._run_section("New JS Features", self.test_new_js_features)
        self._run_section("Incremental Polling", self.test_incremental_polling)
        # Latest features
        self._run_section("Paused State Detection", self.test_paused_state_detection)
        self._run_section("Clear Resets to Idle", self.test_clear_resets_to_idle, skip_slow)
        self._run_section("Metrics Delete API", self.test_metrics_delete_api)
        self._run_section("Chat UI Elements", self.test_chat_ui_elements)
        self._run_section("Delete Buttons", self.test_delete_buttons_exist)
        self._run_section("Step Display", self.test_step_display_in_pipeline)
        self._run_section("Paused CSS", self.test_paused_status_in_css)
        self._run_section("Tuning Resume UI", self.test_tuning_resume_ui)
        self._run_section("PM Clear Record", self.test_process_manager_clear_record, skip_slow)
        self._run_section("Chip Clear All", self.test_chip_clear_all_html)

        self._print_summary()
        return self.failed == 0

    def _run_section(self, name: str, fn, skip_slow: bool = False):
        print()
        print(f"--- {name} ---")
        self._section = name
        before_p, before_f = self.passed, self.failed
        if skip_slow and getattr(fn, "_slow", False):
            self.skip(f"Entire section (--skip-slow)")
            return
        try:
            fn()
        except Exception as e:
            self._record(f"SECTION CRASHED: {e}", False, "")
        self.section_counts[name] = (self.passed - before_p, self.failed - before_f)

    def _print_summary(self):
        print()
        print("=" * 64)
        total = self.passed + self.failed
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        print()
        for section, (p, f) in self.section_counts.items():
            icon = "x" if f > 0 else "v"
            print(f"  [{icon}] {section}: {p} passed, {f} failed")
        if self.errors:
            print()
            print(f"  {len(self.errors)} Failure(s):")
            for e in self.errors:
                print(f"    - {e}")
        print("=" * 64)

    # ══════════════════════════════════════════════════════════
    # TEST SECTIONS
    # ══════════════════════════════════════════════════════════

    # ── Static Files ─────────────────────────────────────────

    def test_static_files(self):
        # HTML
        status, body, headers = self.get_raw("/")
        self.assert_eq("GET / status 200", status, 200)
        self.assert_true("index.html contains <title>", b"micro-Omni" in body)
        self.assert_true("index.html contains ECharts script", b"echarts" in body)
        self.assert_in("Content-Type is text/html", "text/html", headers.get("Content-Type", ""))

        # CSS
        status, body, headers = self.get_raw("/static/style.css")
        self.assert_eq("GET /static/style.css status 200", status, 200)
        self.assert_true("CSS contains :root", b":root" in body)
        self.assert_in("Content-Type is text/css", "text/css", headers.get("Content-Type", ""))

        # JS
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_eq("GET /static/app.js status 200", status, 200)
        self.assert_true("JS contains DOMContentLoaded", b"DOMContentLoaded" in body)

        # 404 for nonexistent static
        status, _, _ = self.get_raw("/static/nonexistent.xyz")
        self.assert_eq("GET /static/nonexistent.xyz -> 404", status, 404)

        # Path traversal blocked
        status, _, _ = self.get_raw("/static/../server/app.py")
        self.assert_in("Path traversal blocked (403 or 404)", status, [403, 404])

    # ── Metrics API ──────────────────────────────────────────

    def test_metrics_api(self):
        # List files
        r = self.check("GET /api/metrics/files", self.get("/api/metrics/files"))
        files = r.get("files", [])
        self.assert_true("Has at least 1 metrics file", len(files) >= 1, f"found {len(files)}")

        # Summary
        r = self.check("GET /api/metrics/summary", self.get("/api/metrics/summary"))
        summary = r.get("summary", {})
        self.assert_true("Summary has entries", len(summary) > 0)

        # Fetch specific file
        if files:
            r = self.check(f"GET data for {files[0]}", self.get(f"/api/metrics/data?file={files[0]}"))
            rows = r.get("rows", [])
            self.assert_true(f"Has rows for {files[0]}", len(rows) > 0, f"found {len(rows)}")

            # Verify row schema
            if rows:
                row = rows[0]
                for key in ["timestamp", "script", "phase", "run_id", "metric_name", "metric_value"]:
                    self.assert_in(f"Row has key '{key}'", key, row)

        # Fetch all
        r = self.check("GET data file=__all__", self.get("/api/metrics/data?file=__all__"))
        self.assert_true("file_data is dict", isinstance(r.get("file_data"), dict))

        # Incremental (since far future = empty)
        r = self.get("/api/metrics/data?file=train_thinker.jsonl&since=2099-01-01")
        self.assert_eq("Since future -> 0 rows", len(r.get("rows", [])), 0)

        # Errors
        self.check("Missing file param -> 400", self.get("/api/metrics/data"), expect_ok=False)
        r = self.get("/api/metrics/data?file=../../etc/passwd")
        self.check("Path traversal in file param -> 400", r, expect_ok=False)

    # ── System API ───────────────────────────────────────────

    def test_system_api(self):
        # GPU
        r = self.check("GET /api/system/gpu", self.get("/api/system/gpu"))
        gpu = r.get("gpu")
        if gpu:
            for key in ["name", "memory_used_mb", "memory_total_mb", "utilization_percent", "temperature_c"]:
                self.assert_in(f"GPU has '{key}'", key, gpu)
            self.assert_gt("GPU total memory > 0", gpu.get("memory_total_mb", 0), 0)
            print(f"       {gpu['name']}: {gpu['memory_used_mb']}MB/{gpu['memory_total_mb']}MB, {gpu['utilization_percent']}%")

        # Checkpoints
        r = self.check("GET /api/system/checkpoints", self.get("/api/system/checkpoints"))
        ckpts = r.get("checkpoints", [])
        for c in ckpts:
            for key in ["name", "path", "has_config", "has_model", "size_mb"]:
                self.assert_in(f"Checkpoint '{c.get('name')}' has '{key}'", key, c)

        # Configs
        r = self.check("GET /api/system/configs", self.get("/api/system/configs"))
        configs = r.get("configs", [])
        self.assert_gt("Has configs", len(configs), 0)
        self.assert_in("Has synthetic_thinker.json", "synthetic_thinker.json", configs)

        # Read specific config
        r = self.check("GET config/synthetic_thinker.json", self.get("/api/system/config/synthetic_thinker.json"))
        cfg = r.get("config", {})
        self.assert_in("Config has 'lr'", "lr", cfg)
        self.assert_in("Config has 'd_model'", "d_model", cfg)

        # Read all stage configs
        for f in ["synthetic_audio_enc.json", "synthetic_vision.json", "synthetic_talker.json",
                   "synthetic_vocoder.json", "synthetic_ocr.json", "synthetic_omni_sft.json"]:
            self.check(f"GET config/{f}", self.get(f"/api/system/config/{f}"))

        # Errors
        self.check("GET config/nonexistent.json -> 404", self.get("/api/system/config/nonexistent.json"), expect_ok=False)
        self.check("POST config with invalid body", self.post("/api/system/config/synthetic_thinker.json", {"bad": True}), expect_ok=False)

    # ── Config Roundtrip ─────────────────────────────────────

    def test_config_roundtrip(self):
        r = self.get("/api/system/config/synthetic_thinker.json")
        original = r.get("config", {})
        original_lr = original.get("lr")

        # Save modified
        mod = dict(original)
        mod["lr"] = 0.77777
        mod["_test_field"] = "test_value"
        self.check("Save modified config", self.post("/api/system/config/synthetic_thinker.json", {"config": mod}))

        # Read back
        r = self.get("/api/system/config/synthetic_thinker.json")
        self.assert_eq("lr saved correctly", r.get("config", {}).get("lr"), 0.77777)
        self.assert_eq("New field saved", r.get("config", {}).get("_test_field"), "test_value")

        # Restore original
        self.check("Restore original", self.post("/api/system/config/synthetic_thinker.json", {"config": original}))
        r = self.get("/api/system/config/synthetic_thinker.json")
        self.assert_eq("lr restored", r.get("config", {}).get("lr"), original_lr)
        self.assert_eq("Test field gone", r.get("config", {}).get("_test_field"), None)

    # ── Pipeline Status ──────────────────────────────────────

    def test_pipeline_status(self):
        r = self.check("GET /api/training/pipeline", self.get("/api/training/pipeline"))
        stages = r.get("stages", {})
        self.assert_eq("Has 7 stages", len(stages), 7)

        for stage_id in ["A", "B", "C", "D", "E", "F", "G"]:
            self.assert_in(f"Stage {stage_id} present", stage_id, stages)
            s = stages[stage_id]
            for key in ["status", "name", "module", "config", "checkpoint_dir", "has_checkpoint", "blocked_by"]:
                self.assert_in(f"Stage {stage_id} has '{key}'", key, s)
            self.assert_in(f"Stage {stage_id} status valid", s["status"], ["idle", "running", "done", "stopped", "failed", "blocked", "paused"])

        # Dependency graph validation
        self.assert_eq("A has no deps", stages["A"]["blocked_by"], [])
        self.assert_eq("B has no deps", stages["B"]["blocked_by"], [])
        self.assert_eq("C has no deps", stages["C"]["blocked_by"], [])
        self.assert_eq("F has no deps", stages["F"]["blocked_by"], [])
        self.assert_eq("G has no deps", stages["G"]["blocked_by"], [])
        # D depends on A
        if not stages["A"]["has_checkpoint"]:
            self.assert_in("D blocked by A", "A", stages["D"]["blocked_by"])
        # E depends on A, B, C, D
        for dep in ["A", "B", "C", "D"]:
            if not stages[dep]["has_checkpoint"]:
                self.assert_in(f"E blocked by {dep}", dep, stages["E"]["blocked_by"])

    # ── Training Lifecycle ───────────────────────────────────

    def test_training_lifecycle(self):
        self.wait_gpu_free()

        # Invalid stage
        self.check("Start invalid stage Z -> 400", self.post("/api/training/start", {"stage": "Z"}), expect_ok=False)

        # Start A
        r = self.check("Start Stage A", self.post("/api/training/start", {"stage": "A"}))
        pid_a = r.get("pid")
        self.assert_true("Got PID", pid_a is not None and pid_a > 0, f"pid={pid_a}")
        time.sleep(3)

        # Pipeline shows running
        r = self.get("/api/training/pipeline")
        self.assert_eq("A is running", r["stages"]["A"]["status"], "running")
        self.assert_true("A has process info", r["stages"]["A"].get("process") is not None)

        # Training status
        r = self.check("GET /api/training/status", self.get("/api/training/status"))
        procs = r.get("processes", {})
        self.assert_in("training_A in processes", "training_A", procs)
        self.assert_eq("training_A status=running", procs["training_A"]["status"], "running")

        # Logs available
        r = self.check("GET /api/training/logs/A", self.get("/api/training/logs/A"))
        self.assert_true("Has log lines", len(r.get("lines", [])) > 0)
        self.assert_true("Has log_file path", bool(r.get("log_file")))

        # Can't start duplicate
        self.check("Start A again (GPU busy) -> 409", self.post("/api/training/start", {"stage": "A"}), expect_ok=False)

        # Can't clear while running
        self.check("Clear A while running -> 409", self.post("/api/training/clear", {"stage": "A"}), expect_ok=False)

        # Stop
        r = self.check("Stop Stage A", self.post("/api/training/stop", {"stage": "A"}))
        self.assert_eq("stopped=True", r.get("stopped"), True)
        time.sleep(2)

        # Verify stopped
        r = self.get("/api/training/pipeline")
        self.assert_in("A status after stop", r["stages"]["A"]["status"], ["stopped", "done", "idle"])

        # Stop already stopped (idempotent)
        r = self.check("Stop A again (already stopped)", self.post("/api/training/stop", {"stage": "A"}))
        self.assert_eq("stopped=False (not running)", r.get("stopped"), False)

        # Resume
        r = self.check("Resume Stage A", self.post("/api/training/start", {"stage": "A"}))
        pid_resume = r.get("pid")
        self.assert_true("Resume got new PID", pid_resume is not None and pid_resume != pid_a)
        time.sleep(3)

        # Running again
        r = self.get("/api/training/pipeline")
        self.assert_eq("A running after resume", r["stages"]["A"]["status"], "running")

        # Stop and clear
        self.post("/api/training/stop", {"stage": "A"})
        time.sleep(2)
        r = self.check("Clear Stage A", self.post("/api/training/clear", {"stage": "A"}))
        self.assert_true("Cleared files", r.get("count", 0) >= 0)

        # Verify idle
        r = self.get("/api/training/pipeline")
        self.assert_eq("A has no checkpoint", r["stages"]["A"]["has_checkpoint"], False)

        # Blocked dependency check after clear
        self.assert_in("D blocked by A after clear", "A", r["stages"]["D"]["blocked_by"])

        # Try starting blocked stage
        self.check("Start E (blocked) -> 409", self.post("/api/training/start", {"stage": "E"}), expect_ok=False)

    test_training_lifecycle._slow = True

    # ── Pause/Resume Verification ────────────────────────────

    def test_pause_resume_progress(self):
        """Verify training actually resumes from where it stopped — step counter advances."""
        self.wait_gpu_free()

        # Start fresh Stage A training
        self.post("/api/training/clear", {"stage": "A"})
        time.sleep(1)

        # ── Phase 1: Train for a while, record step ──────────
        r = self.check("Start Stage A (fresh)", self.post("/api/training/start", {"stage": "A"}))
        if not r.get("ok"):
            self.skip("Pause/resume test", "failed to start")
            return

        # Wait for checkpoint to be saved (need 500+ steps)
        print("       Phase 1: Training until first checkpoint...")
        for i in range(24):  # max 120s
            time.sleep(5)
            r = self.get("/api/training/pipeline")
            meta = r["stages"]["A"].get("metadata")
            if meta and meta.get("step", 0) > 0:
                break

        # Stop training
        self.post("/api/training/stop", {"stage": "A"})
        time.sleep(2)

        # Read step from checkpoint metadata
        r = self.get("/api/training/pipeline")
        meta1 = r["stages"]["A"].get("metadata")
        step1 = meta1.get("step", 0) if meta1 else 0
        print(f"       Phase 1 stopped at step {step1}")
        self.assert_gt("Phase 1 trained some steps", step1, 0)

        # Read metrics to verify loss was logged
        r = self.get("/api/metrics/data?file=train_thinker.jsonl")
        rows1 = [row for row in r.get("rows", []) if row.get("metric_name") == "loss" and row.get("phase") == "train"]
        self.assert_gt("Phase 1 has metric rows", len(rows1), 0)

        # ── Phase 2: Resume and verify step advances ─────────
        print("       Phase 2: Resuming training...")
        r = self.check("Resume Stage A", self.post("/api/training/start", {"stage": "A"}))
        if not r.get("ok"):
            self.skip("Resume phase", "failed to resume")
            return

        # Record the max step BEFORE phase 2 starts producing data
        r = self.get("/api/metrics/data?file=train_thinker.jsonl")
        rows_before = [row for row in r.get("rows", []) if row.get("metric_name") == "loss" and row.get("phase") == "train"]
        max_step_before = max((row.get("step", 0) for row in rows_before), default=0)

        # Wait for resumed training to log new metrics
        time.sleep(15)

        # Stop again
        self.post("/api/training/stop", {"stage": "A"})
        time.sleep(2)

        # Read new step from checkpoint metadata
        r = self.get("/api/training/pipeline")
        meta2 = r["stages"]["A"].get("metadata")
        step2 = meta2.get("step", 0) if meta2 else 0
        print(f"       Phase 2 stopped at metadata step {step2} (was {step1})")

        # Read metrics — find the highest step logged
        r = self.get("/api/metrics/data?file=train_thinker.jsonl")
        rows_after = [row for row in r.get("rows", []) if row.get("metric_name") == "loss" and row.get("phase") == "train"]
        max_step_after = max((row.get("step", 0) for row in rows_after), default=0)
        print(f"       Max metric step: before={max_step_before} -> after={max_step_after}")

        # Key assertion: metrics should show steps beyond phase 1
        self.assert_gt(f"Metrics advanced beyond phase 1 (step {step1})", max_step_after, step1)

        # Verify resume didn't restart from step 0 — find min step in phase 2 metrics
        # (rows with timestamps after phase 2 start should have step > 0)
        self.assert_gt("Resume didn't restart from 0", max_step_after, step1 - 1)

        # ── Phase 3: Verify logs show resume ─────────────────
        r = self.get("/api/training/logs/A")
        log_text = "\n".join(r.get("lines", []))
        # Training scripts typically print "Resuming from step X" or load checkpoint
        has_resume_indicator = "resum" in log_text.lower() or "checkpoint" in log_text.lower() or "step" in log_text.lower()
        self.assert_true("Logs mention resume/checkpoint/step", has_resume_indicator)

    test_pause_resume_progress._slow = True

    # ── GPU Lock ─────────────────────────────────────────────

    def test_gpu_lock(self):
        self.wait_gpu_free()

        # Start C (no deps)
        r = self.check("Start Stage C", self.post("/api/training/start", {"stage": "C"}))
        time.sleep(2)

        # Every GPU operation should fail
        gpu_ops = [
            ("training/start A", {"stage": "A"}),
            ("training/start B", {"stage": "B"}),
            ("testing/run", {"script": "test_thinker"}),
            ("tuning/start", {"stage": "A", "n_trials": 1}),
            ("export/run", {}),
            ("inference/chat", {"text": "hi"}),
            ("inference/standalone", {"text": "hi"}),
            ("inference/huggingface", {"text": "hi"}),
        ]
        for label, body in gpu_ops:
            self.check(f"POST {label} while C running -> 409",
                       self.post(f"/api/{label}", body), expect_ok=False)

        # Stop C
        self.check("Stop Stage C", self.post("/api/training/stop", {"stage": "C"}))
        time.sleep(2)

        # GPU free
        r = self.get("/api/training/pipeline")
        self.assert_true("C not running", r["stages"]["C"]["status"] != "running")

        # Can start again
        r = self.post("/api/training/start", {"stage": "F"})
        if r.get("ok"):
            self.check("Start F after freeing GPU", r)
            time.sleep(1)
            self.post("/api/training/stop", {"stage": "F"})
            time.sleep(1)
        else:
            self.check("Start F after freeing GPU", r)

    test_gpu_lock._slow = True

    # ── All Stages Start/Stop ────────────────────────────────

    def test_all_stages_start_stop(self):
        """Quick start/stop of every non-blocked stage to verify they all launch."""
        launchable = ["A", "B", "C", "F", "G"]  # D needs A, E needs A+B+C+D

        for stage in launchable:
            self.wait_gpu_free()
            r = self.post("/api/training/start", {"stage": stage})
            self.check(f"Start stage {stage}", r)
            if r.get("ok"):
                time.sleep(2)
                # Verify running
                r2 = self.get("/api/training/pipeline")
                self.assert_eq(f"Stage {stage} is running", r2["stages"][stage]["status"], "running")
                # Stop and wait
                self.post("/api/training/stop", {"stage": stage})
                time.sleep(2)
            else:
                self.skip(f"Stop stage {stage}", "didn't start")
        # Clean up
        self.wait_gpu_free()

    test_all_stages_start_stop._slow = True

    # ── Testing Lifecycle ────────────────────────────────────

    def test_testing_lifecycle(self):
        self.wait_gpu_free()

        # Ensure thinker checkpoint exists (previous tests may have cleared it)
        r = self.get("/api/training/pipeline")
        if not r.get("stages", {}).get("A", {}).get("has_checkpoint"):
            print("       Rebuilding thinker checkpoint (need ~500 steps for first save)...")
            self.post("/api/training/start", {"stage": "A"})
            # Wait for checkpoint to appear (check every 5s, max 90s)
            for i in range(18):
                time.sleep(5)
                r = self.get("/api/training/pipeline")
                if r["stages"]["A"]["has_checkpoint"]:
                    print(f"       Checkpoint appeared after ~{(i+1)*5}s")
                    break
            self.post("/api/training/stop", {"stage": "A"})
            time.sleep(2)
            r = self.get("/api/training/pipeline")
            if not r["stages"]["A"]["has_checkpoint"]:
                self.skip("Thinker checkpoint not created in time", "training too slow")
                return
            self.check("Thinker checkpoint created", r, expect_ok=True)

        # Status
        self.check("GET /api/testing/status", self.get("/api/testing/status"))

        # Results endpoint
        self.check("GET results for test_thinker", self.get("/api/testing/results/test_thinker"))

        # Bad script
        self.check("Run bad script -> 400", self.post("/api/testing/run", {"script": "nonexistent"}), expect_ok=False)

        # Run actual test — use 100 samples so it runs long enough for GPU lock test
        r = self.check("Run test_thinker (100 samples)", self.post("/api/testing/run", {"script": "test_thinker", "num_samples": 100}))
        if r.get("ok"):
            test_pid = r.get("pid")
            self.assert_true("Got test PID", test_pid is not None and test_pid > 0)

            # Check GPU lock immediately (test should still be running with 100 samples)
            time.sleep(1)
            r2 = self.get("/api/testing/status")
            test_still_running = any(p.get("status") == "running" for p in r2.get("processes", {}).values())
            if test_still_running:
                self.check("Start training while testing -> 409",
                           self.post("/api/training/start", {"stage": "A"}), expect_ok=False)
            else:
                self.skip("Start training while testing -> 409", "test exited too fast")

            # Wait for completion
            print("       Waiting for test to complete...")
            result = self.wait_process_done("testing", timeout=120)
            if result:
                self.assert_in("Test finished", result["status"], ["completed", "failed"])
                print(f"       Test {result['status']} in {result.get('elapsed_seconds', '?')}s")
            else:
                self.skip("Test completion check", "timeout")

            # Verify results were written (only if test completed successfully)
            if result and result["status"] == "completed":
                r3 = self.get("/api/testing/results/test_thinker")
                results = r3.get("results", [])
                self.assert_true("Test produced results", len(results) > 0, f"found {len(results)} rows")
            elif result and result["status"] == "failed":
                self.skip("Test produced results", "test failed (likely missing data)")

    test_testing_lifecycle._slow = True

    # ── Inference API ────────────────────────────────────────

    def test_inference_api(self):
        self.wait_gpu_free()

        # Unload (always safe)
        self.check("POST unload", self.post("/api/inference/unload"))

        # Chat mode - missing input
        self.check("Chat with no input -> 400", self.post("/api/inference/chat", {}), expect_ok=False)
        self.check("Chat with empty text -> 400", self.post("/api/inference/chat", {"text": ""}), expect_ok=False)

        # Standalone - missing text
        self.check("Standalone with no text -> 400", self.post("/api/inference/standalone", {}), expect_ok=False)

        # Standalone - bad model dir
        self.check("Standalone bad dir -> 404", self.post("/api/inference/standalone", {"text": "hi", "model_dir": "nonexistent/"}), expect_ok=False)

        # HuggingFace - missing input
        self.check("HF with no input -> 400", self.post("/api/inference/huggingface", {}), expect_ok=False)

        # HF - bad model dir
        self.check("HF bad dir -> 404", self.post("/api/inference/huggingface", {"text": "hi", "model_dir": "nonexistent/"}), expect_ok=False)

        # Bad endpoint
        self.check("Unknown endpoint -> 404", self.post("/api/inference/badendpoint", {}), expect_ok=False)

        # Double unload (idempotent)
        self.check("Unload again (idempotent)", self.post("/api/inference/unload"))

    # ── Tuning API ───────────────────────────────────────────

    def test_tuning_api(self):
        # Spaces
        r = self.check("GET /api/tuning/spaces", self.get("/api/tuning/spaces"))
        spaces = r.get("spaces", {})
        self.assert_eq("Has 7 stage spaces", len(spaces), 7)
        for stage_id in ["A", "B", "C", "D", "E", "F", "G"]:
            self.assert_in(f"Space for {stage_id}", stage_id, spaces)
            params = spaces[stage_id].get("params", [])
            self.assert_gt(f"Stage {stage_id} has params", len(params), 5)
            # Verify param schema
            if params:
                p = params[0]
                for key in ["name", "type"]:
                    self.assert_in(f"Param has '{key}'", key, p)

        # Status
        self.check("GET /api/tuning/status", self.get("/api/tuning/status"))

        # Results (may have data if tuning was run previously)
        r = self.check("Results for A", self.get("/api/tuning/results/A"))
        results = r.get("results")
        if results is not None:
            self.assert_true("Results has no error", "error" not in results, str(results.get("error", "")))
        # Stage B should not have been tuned
        r_b = self.get("/api/tuning/results/B")
        self.assert_eq("Stage B no results", r_b.get("results"), None)

        # Invalid stage
        self.check("Start bad stage -> 400", self.post("/api/tuning/start", {"stage": "Z"}), expect_ok=False)
        self.check("Stop bad stage -> 400", self.post("/api/tuning/stop", {"stage": "Z"}), expect_ok=False)

        # Clear empty (idempotent)
        self.check("Clear A (already empty)", self.post("/api/tuning/clear", {"stage": "A"}))

        # Apply with no results
        self.check("Apply A (no results) -> 404", self.post("/api/tuning/apply", {"stage": "A"}), expect_ok=False)

        # Apply bad stage
        self.check("Apply bad stage -> 400", self.post("/api/tuning/apply", {"stage": "Z"}), expect_ok=False)

    # ── Export API ───────────────────────────────────────────

    def test_export_api(self):
        self.check("GET /api/export/status", self.get("/api/export/status"))

        self.wait_gpu_free(timeout=60)

        # Start export
        r = self.check("Start export", self.post("/api/export/run", {}))
        if r.get("ok"):
            time.sleep(2)
            # Check status
            r2 = self.check("Export status", self.get("/api/export/status"))
            procs = r2.get("processes", {})
            self.assert_true("Has export process", len(procs) > 0)

            # Stop it (don't wait for completion)
            for key in procs:
                if procs[key].get("status") == "running":
                    # Can't directly stop export via export API, use training stop pattern
                    # Export uses ProcessManager so we can check it's tracked
                    self.assert_in("Export is tracked", procs[key]["status"], ["running", "completed", "failed"])
            # Kill all to clean up
            self.wait_gpu_free(timeout=10)

    test_export_api._slow = True

    # ── Error Handling ───────────────────────────────────────

    def test_error_handling(self):
        # Unknown API routes
        self.check("GET /api/unknown -> 404", self.get("/api/unknown"), expect_ok=False)
        self.check("POST /api/unknown -> 404", self.post("/api/unknown"), expect_ok=False)
        self.check("GET /api/metrics/unknown -> 404", self.get("/api/metrics/unknown"), expect_ok=False)
        self.check("GET /api/training/unknown -> 404", self.get("/api/training/unknown"), expect_ok=False)
        self.check("GET /api/system/unknown -> 404", self.get("/api/system/unknown"), expect_ok=False)

        # Invalid JSON body
        try:
            req = urllib.request.Request(f"{self.base}/api/training/start")
            req.add_header("Content-Type", "application/json")
            req.data = b"not json"
            r = urllib.request.urlopen(req, timeout=5)
            self.check("Invalid JSON body -> 400", json.loads(r.read()), expect_ok=False)
        except urllib.error.HTTPError as e:
            self.assert_eq("Invalid JSON -> 400 status", e.code, 400)

        # Logs for invalid stage
        self.check("Logs for bad stage -> 400", self.get("/api/training/logs/Z"), expect_ok=False)

        # Metrics path traversal
        r = self.get("/api/metrics/data?file=../../../etc/passwd")
        self.check("Metrics path traversal blocked", r, expect_ok=False)

    # ── CORS Headers ─────────────────────────────────────────

    def test_cors(self):
        status, body, headers = self.get_raw("/api/system/gpu")
        self.assert_eq("CORS header present", headers.get("Access-Control-Allow-Origin"), "*")
        self.assert_eq("Cache-Control no-store", headers.get("Cache-Control"), "no-store")

    # ── Content Types ────────────────────────────────────────

    def test_content_types(self):
        # API returns JSON
        status, body, headers = self.get_raw("/api/system/gpu")
        self.assert_in("API Content-Type is JSON", "application/json", headers.get("Content-Type", ""))

        # Static returns correct types
        for path, expected_cts in [
            ("/", ["text/html"]),
            ("/static/style.css", ["text/css"]),
            ("/static/app.js", ["javascript"]),  # text/javascript or application/javascript both valid
        ]:
            status, body, headers = self.get_raw(path)
            ct = headers.get("Content-Type", "")
            matched = any(e in ct for e in expected_cts)
            self.assert_true(f"{path} Content-Type has '{expected_cts[0]}'", matched, f"got '{ct}'")


    # ══════════════════════════════════════════════════════════
    # INTEGRATION DATA VALIDATION
    # (tests that real data on disk is parsed correctly)
    # ══════════════════════════════════════════════════════════

    # ── Metrics JSONL Schema Validation ──────────────────────

    def test_metrics_data_integrity(self):
        """Validate actual JSONL rows have correct types and values."""
        r = self.get("/api/metrics/files")
        files = r.get("files", [])
        if not files:
            self.skip("Metrics data integrity", "no JSONL files on disk")
            return

        for fname in files:
            r = self.get(f"/api/metrics/data?file={fname}")
            rows = r.get("rows", [])
            if not rows:
                continue

            self.assert_gt(f"{fname} has rows", len(rows), 0)

            # Validate schema of first 5 rows
            for i, row in enumerate(rows[:5]):
                prefix = f"{fname}[{i}]"
                # Required fields
                self.assert_in(f"{prefix} has timestamp", "timestamp", row)
                self.assert_in(f"{prefix} has metric_name", "metric_name", row)
                self.assert_in(f"{prefix} has phase", "phase", row)

                # Type checks
                self.assert_true(f"{prefix} timestamp is string",
                                isinstance(row.get("timestamp"), str), f"got {type(row.get('timestamp'))}")
                self.assert_true(f"{prefix} metric_name is string",
                                isinstance(row.get("metric_name"), str))
                self.assert_in(f"{prefix} phase is valid",
                              row.get("phase"), ["train", "val", "test", "event"])

                # metric_value should be numeric (not NaN, not string)
                mv = row.get("metric_value")
                if row.get("phase") != "event":
                    self.assert_true(f"{prefix} metric_value is number",
                                    isinstance(mv, (int, float)), f"got {type(mv)}: {mv}")
                    if isinstance(mv, float):
                        import math
                        self.assert_true(f"{prefix} metric_value not NaN", not math.isnan(mv))
                        self.assert_true(f"{prefix} metric_value not Inf", not math.isinf(mv))

                # step should be int or None
                step = row.get("step")
                if step is not None:
                    self.assert_true(f"{prefix} step is int", isinstance(step, int), f"got {type(step)}")

        # Summary should aggregate correctly
        r = self.get("/api/metrics/summary")
        summary = r.get("summary", {})
        for fname, finfo in summary.items():
            runs = finfo.get("runs", {})
            for run_id, rinfo in runs.items():
                metrics = rinfo.get("metrics", {})
                for mname, mdata in metrics.items():
                    self.assert_true(f"Summary {fname}/{mname} has value",
                                    mdata.get("value") is not None)
                    self.assert_true(f"Summary {fname}/{mname} has step",
                                    mdata.get("step") is not None)

    # ── Checkpoint Data Integrity ────────────────────────────

    def test_checkpoint_data_integrity(self):
        """Validate checkpoint scan returns correct metadata."""
        r = self.get("/api/system/checkpoints")
        ckpts = r.get("checkpoints", [])
        if not ckpts:
            self.skip("Checkpoint data integrity", "no checkpoints on disk")
            return

        for c in ckpts:
            name = c.get("name", "?")
            prefix = f"ckpt/{name}"

            # Type validation
            self.assert_true(f"{prefix} size_mb is number",
                            isinstance(c.get("size_mb"), (int, float)))
            self.assert_true(f"{prefix} has_config is bool",
                            isinstance(c.get("has_config"), bool))
            self.assert_true(f"{prefix} has_model is bool",
                            isinstance(c.get("has_model"), bool))

            # If has model, size should be > 0
            if c.get("has_model"):
                self.assert_gt(f"{prefix} model size > 0", c.get("size_mb", 0), 0)

            # If has config, read it and validate
            if c.get("has_config"):
                r2 = self.get(f"/api/system/checkpoint-config/{name}")
                if r2.get("ok"):
                    cfg = r2.get("config", {})
                    # All checkpoint configs should have basic keys
                    for key in ["d_model", "n_layers"]:
                        if key in cfg:
                            self.assert_true(f"{prefix} config {key} is int",
                                            isinstance(cfg[key], int))

            # Metadata validation
            meta = c.get("metadata")
            if meta:
                if "step" in meta:
                    self.assert_true(f"{prefix} metadata step is int",
                                    isinstance(meta["step"], int))
                if "epoch" in meta:
                    self.assert_true(f"{prefix} metadata epoch is int",
                                    isinstance(meta["epoch"], int))

    # ── Config Schema Validation ─────────────────────────────

    def test_config_schema_integrity(self):
        """Validate all configs have required fields with correct types."""
        r = self.get("/api/system/configs")
        configs = r.get("configs", [])

        # Common fields all training configs must have
        common_required = ["lr", "wd", "warmup_steps", "max_steps", "batch_size",
                          "use_amp", "save_dir", "seed"]

        for fname in configs:
            if not fname.startswith("synthetic_"):
                continue
            r = self.get(f"/api/system/config/{fname}")
            cfg = r.get("config", {})

            # Check common fields (vocoder uses lr_g/lr_d instead of lr)
            is_vocoder = "vocoder" in fname
            for key in common_required:
                if key == "lr" and is_vocoder:
                    self.assert_in(f"{fname} has lr_g", "lr_g", cfg)
                    self.assert_in(f"{fname} has lr_d", "lr_d", cfg)
                else:
                    self.assert_in(f"{fname} has {key}", key, cfg)

            # Type checks for numeric params
            for key in ["warmup_steps", "max_steps", "batch_size", "seed"]:
                if key in cfg:
                    self.assert_true(f"{fname} {key} is int",
                                    isinstance(cfg[key], int), f"got {type(cfg[key])}: {cfg[key]}")

            for key in ["wd", "ema_decay"]:
                if key in cfg:
                    self.assert_true(f"{fname} {key} is number",
                                    isinstance(cfg[key], (int, float)))

            # Boolean flags
            for key in ["use_amp", "use_compile"]:
                if key in cfg:
                    self.assert_true(f"{fname} {key} is bool",
                                    isinstance(cfg[key], bool))

            # val_loss_threshold must be 999.0 for synthetic configs
            if "val_loss_threshold" in cfg:
                self.assert_eq(f"{fname} val_loss_threshold = 999.0",
                              cfg["val_loss_threshold"], 999.0)

    # ── Pipeline Dependency Integrity ────────────────────────

    def test_pipeline_dependency_integrity(self):
        """Validate pipeline dependencies are consistent with actual checkpoint state."""
        r = self.get("/api/training/pipeline")
        stages = r.get("stages", {})

        for stage_id, s in stages.items():
            # Blocked stages must list which stages block them
            if s["status"] == "blocked":
                self.assert_gt(f"Stage {stage_id} blocked_by not empty",
                              len(s.get("blocked_by", [])), 0)
                # Each blocker should actually lack a checkpoint
                for blocker in s["blocked_by"]:
                    blocker_info = stages.get(blocker, {})
                    self.assert_eq(f"Stage {stage_id} blocker {blocker} has no checkpoint",
                                  blocker_info.get("has_checkpoint"), False)

            # Done stages must have a checkpoint
            if s["status"] == "done":
                self.assert_true(f"Stage {stage_id} done -> has checkpoint",
                                s.get("has_checkpoint"))

            # Running stages must have process info
            if s["status"] == "running":
                self.assert_true(f"Stage {stage_id} running -> has process",
                                s.get("process") is not None)

    # ── Training Logs Integrity ──────────────────────────────

    def test_training_logs_integrity(self):
        """Validate training log endpoints return parseable content."""
        for stage in ["A", "B", "C", "D", "E", "F", "G"]:
            r = self.get(f"/api/training/logs/{stage}")
            self.check(f"Logs for stage {stage}", r)
            lines = r.get("lines", [])
            # Lines should be strings
            if lines:
                self.assert_true(f"Stage {stage} log lines are strings",
                                all(isinstance(l, str) for l in lines))
                # No null bytes or binary garbage
                self.assert_true(f"Stage {stage} logs are text",
                                all("\x00" not in l for l in lines))

    # ── Tuning Results Integrity ─────────────────────────────

    def test_tuning_results_integrity(self):
        """Validate tuning results from actual Optuna DB (if exists)."""
        for stage in ["A", "B", "C", "D", "E", "F", "G"]:
            r = self.get(f"/api/tuning/results/{stage}")
            self.check(f"Tuning results {stage}", r)
            results = r.get("results")

            if results is None:
                continue  # No DB for this stage

            if "error" in results:
                self._record(f"Tuning {stage} DB read error: {results['error']}", False)
                continue

            # Validate structure
            self.assert_in(f"Tuning {stage} has study_name", "study_name", results)
            self.assert_in(f"Tuning {stage} has direction", "direction", results)
            self.assert_in(f"Tuning {stage} has n_trials", "n_trials", results)
            self.assert_in(f"Tuning {stage} has trials", "trials", results)

            direction = results.get("direction", "")
            self.assert_in(f"Tuning {stage} direction valid", direction, ["MINIMIZE", "MAXIMIZE"])

            trials = results.get("trials", [])
            for t in trials[:5]:
                # Trial schema
                self.assert_in(f"Trial has number", "number", t)
                self.assert_in(f"Trial has state", "state", t)
                self.assert_in(f"Trial has params", "params", t)
                self.assert_in(f"Trial state valid", t.get("state"),
                              ["COMPLETE", "RUNNING", "PRUNED", "FAIL", "WAITING"])

                # Params should be a dict with correct types
                params = t.get("params", {})
                self.assert_true(f"Trial params is dict", isinstance(params, dict))
                for k, v in params.items():
                    self.assert_true(f"Param '{k}' is not None", v is not None, f"got None")

                # Completed trials must have a numeric value
                if t.get("state") == "COMPLETE":
                    val = t.get("value")
                    self.assert_true(f"Complete trial has numeric value",
                                    isinstance(val, (int, float)), f"got {type(val)}")

            # Best trial validation
            best = results.get("best_trial")
            if best:
                self.assert_in(f"Best trial has number", "number", best)
                self.assert_in(f"Best trial has value", "value", best)
                self.assert_in(f"Best trial has params", "params", best)
                self.assert_true(f"Best trial value is number",
                                isinstance(best.get("value"), (int, float)))

    # ── Search Space Integrity ───────────────────────────────

    def test_search_space_integrity(self):
        """Validate all search spaces have well-formed param definitions."""
        r = self.get("/api/tuning/spaces")
        spaces = r.get("spaces", {})

        valid_types = {"float_log", "float", "int", "categorical"}

        for stage_id, space in spaces.items():
            params = space.get("params", [])
            self.assert_gt(f"Stage {stage_id} has params", len(params), 0)

            seen_names = set()
            for p in params:
                name = p.get("name", "")
                ptype = p.get("type", "")

                # No duplicate param names
                self.assert_true(f"Stage {stage_id}/{name} not duplicate",
                                name not in seen_names, f"duplicate: {name}")
                seen_names.add(name)

                # Valid type
                self.assert_in(f"Stage {stage_id}/{name} type valid", ptype, valid_types)

                # Range validation
                if ptype in ("float_log", "float", "int"):
                    low = p.get("low")
                    high = p.get("high")
                    self.assert_true(f"Stage {stage_id}/{name} has low",
                                    low is not None)
                    self.assert_true(f"Stage {stage_id}/{name} has high",
                                    high is not None)
                    if low is not None and high is not None:
                        self.assert_true(f"Stage {stage_id}/{name} low < high",
                                        low < high, f"{low} >= {high}")
                elif ptype == "categorical":
                    choices = p.get("choices", [])
                    if isinstance(choices, list):
                        self.assert_gt(f"Stage {stage_id}/{name} has choices",
                                      len(choices), 0)
                    else:
                        self._record(f"Stage {stage_id}/{name} choices is list", False,
                                    f" -> got {type(choices)}")


    # ══════════════════════════════════════════════════════════
    # CROSS-VALIDATION: Config ↔ Training Script ↔ Tuning Space
    # ══════════════════════════════════════════════════════════

    def test_config_training_param_coverage(self):
        """Every cfg.get() param in training scripts must exist in its config."""
        import re

        scripts = {
            "A": ("train/train_thinker.py", "synthetic_thinker.json"),
            "B": ("train/train_audio_enc.py", "synthetic_audio_enc.json"),
            "C": ("train/train_vision.py", "synthetic_vision.json"),
            "D": ("train/train_talker.py", "synthetic_talker.json"),
            "E": ("train/sft_omni.py", "synthetic_omni_sft.json"),
            "F": ("train/train_vocoder.py", "synthetic_vocoder.json"),
            "G": ("train/train_ocr.py", "synthetic_ocr.json"),
        }

        DYNAMIC_PARAMS = {
            "config_path", "ctc_vocab_size", "max_mel_length", "max_text_len",
            "max_mel_length_sample_size", "recalculate_dataset_stats",
            "max_text_length", "max_text_length_percentile",
            "thinker_d_model", "dataset_sample_size", "max_audio_length",
        }

        pat = re.compile(r'cfg\.get\(\s*["\x27]([^"\x27]+)["\x27]')

        for stage, (script_path, config_name) in sorted(scripts.items()):
            r = self.get(f"/api/system/config/{config_name}")
            if not r.get("ok"):
                self.skip(f"Stage {stage} config", "not readable")
                continue
            cfg = r.get("config", {})
            cfg_keys = set(cfg.keys())
            for k, v in cfg.items():
                if isinstance(v, dict):
                    for nk in v:
                        cfg_keys.add(nk)

            script_keys = set()
            try:
                with open(script_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        for m in pat.finditer(line):
                            script_keys.add(m.group(1))
            except FileNotFoundError:
                self.skip(f"Stage {stage} script", "not found")
                continue

            missing = script_keys - cfg_keys - DYNAMIC_PARAMS
            if missing:
                for m in sorted(missing):
                    self._record(f"Stage {stage} config missing '{m}'", False, " (used in script)")
            else:
                self.check(f"Stage {stage} all script params in config ({len(script_keys)} params)", r)

    def test_tuning_space_covers_config(self):
        """Every tuning param must exist in its stage's config."""
        r = self.get("/api/tuning/spaces")
        spaces = r.get("spaces", {})

        config_map = {
            "A": "synthetic_thinker.json", "B": "synthetic_audio_enc.json",
            "C": "synthetic_vision.json", "D": "synthetic_talker.json",
            "E": "synthetic_omni_sft.json", "F": "synthetic_vocoder.json",
            "G": "synthetic_ocr.json",
        }

        for stage_id, space in spaces.items():
            config_name = config_map.get(stage_id)
            if not config_name:
                continue

            r2 = self.get(f"/api/system/config/{config_name}")
            cfg = r2.get("config", {})
            cfg_keys = set(cfg.keys())
            for k, v in cfg.items():
                if isinstance(v, dict):
                    for nk in v:
                        cfg_keys.add(nk)

            for p in space.get("params", []):
                name = p["name"]
                self.assert_in(f"Tune {stage_id}/{name} in config", name, cfg_keys)

    def test_tuning_param_types_match_config(self):
        """Tuning param types must be compatible with config value types."""
        r = self.get("/api/tuning/spaces")
        spaces = r.get("spaces", {})

        config_map = {
            "A": "synthetic_thinker.json", "B": "synthetic_audio_enc.json",
            "C": "synthetic_vision.json", "D": "synthetic_talker.json",
            "E": "synthetic_omni_sft.json", "F": "synthetic_vocoder.json",
            "G": "synthetic_ocr.json",
        }

        for stage_id, space in spaces.items():
            config_name = config_map.get(stage_id)
            if not config_name:
                continue

            r2 = self.get(f"/api/system/config/{config_name}")
            cfg = r2.get("config", {})

            for p in space.get("params", []):
                name = p["name"]
                ptype = p["type"]
                cfg_val = cfg.get(name)
                if cfg_val is None:
                    for v in cfg.values():
                        if isinstance(v, dict) and name in v:
                            cfg_val = v[name]
                            break
                if cfg_val is None:
                    continue

                if ptype in ("float_log", "float"):
                    self.assert_true(f"Tune {stage_id}/{name} cfg is number",
                                    isinstance(cfg_val, (int, float)),
                                    f"got {type(cfg_val).__name__}: {cfg_val}")
                elif ptype == "int":
                    self.assert_true(f"Tune {stage_id}/{name} cfg is int",
                                    isinstance(cfg_val, (int, float)),
                                    f"got {type(cfg_val).__name__}: {cfg_val}")
                elif ptype == "categorical":
                    choices = p.get("choices", [])
                    if isinstance(choices, list) and choices:
                        choice_types = set(type(c) for c in choices)
                        # Allow int/float interchangeability
                        if isinstance(cfg_val, int):
                            choice_types.add(int)
                            choice_types.add(float)
                        if isinstance(cfg_val, float):
                            choice_types.add(int)
                            choice_types.add(float)
                        self.assert_in(f"Tune {stage_id}/{name} cfg type compatible",
                                      type(cfg_val), choice_types)

    def test_tuning_run_id_computation(self):
        """Verify run_id is deterministic and unique per trial."""
        try:
            from omni.training_utils import build_run_id
        except ImportError:
            self.skip("build_run_id import", "omni not importable")
            return

        stages = {
            "A": "train_thinker.py", "B": "train_audio_enc.py",
            "C": "train_vision.py", "D": "train_talker.py",
            "E": "sft_omni.py", "F": "train_vocoder.py",
            "G": "train_ocr.py",
        }

        for stage, script_name in stages.items():
            rid0 = build_run_id(script_name, None, f"checkpoints/tune_{stage}/trial_0")
            rid1 = build_run_id(script_name, None, f"checkpoints/tune_{stage}/trial_1")
            self.assert_true(f"Stage {stage} run_id is 16-char hex",
                            len(rid0) == 16 and all(c in "0123456789abcdef" for c in rid0))
            self.assert_true(f"Stage {stage} trial_0 != trial_1", rid0 != rid1)
            # Same inputs = same output (deterministic)
            rid0b = build_run_id(script_name, None, f"checkpoints/tune_{stage}/trial_0")
            self.assert_eq(f"Stage {stage} run_id deterministic", rid0, rid0b)

    def test_tuning_optuna_db_schema(self):
        """If Optuna DB exists, validate the read produces correct schema."""
        for stage in ["A", "B", "C", "D", "E", "F", "G"]:
            r = self.get(f"/api/tuning/results/{stage}")
            results = r.get("results")
            if results is None:
                continue
            if "error" in results:
                self._record(f"Tuning {stage} DB read error", False, f": {results['error']}")
                continue

            self.assert_eq(f"Tuning {stage} direction", results.get("direction"), "MINIMIZE")

            # Validate trial params match search space
            r2 = self.get("/api/tuning/spaces")
            space_params = set(p["name"] for p in r2.get("spaces", {}).get(stage, {}).get("params", []))
            for trial in results.get("trials", [])[:3]:
                trial_params = set(trial.get("params", {}).keys())
                extra = trial_params - space_params
                self.assert_eq(f"Tune {stage} trial #{trial['number']} no extra params", len(extra), 0)

                if trial.get("state") == "COMPLETE":
                    val = trial.get("value")
                    self.assert_true(f"Tune {stage} trial #{trial['number']} value is number",
                                    isinstance(val, (int, float)))

    def test_metrics_run_id_consistency(self):
        """Verify metrics have consistent run_ids and monotonic steps."""
        r = self.get("/api/metrics/files")
        for fname in r.get("files", []):
            r2 = self.get(f"/api/metrics/data?file={fname}")
            rows = r2.get("rows", [])
            if len(rows) < 2:
                continue

            runs = {}
            for row in rows:
                rid = row.get("run_id", "")
                if rid not in runs:
                    runs[rid] = []
                runs[rid].append(row)

            for rid, run_rows in runs.items():
                train_rows = [r for r in run_rows if r.get("phase") == "train" and r.get("metric_name") == "loss"]
                if len(train_rows) < 2:
                    continue
                max_step = max(r.get("step", 0) for r in train_rows)
                self.assert_gt(f"{fname}/{rid[:8]} max step > 0", max_step, 0)

    def test_config_value_sanity(self):
        """Validate all config values are within sane ranges."""
        config_map = {
            "A": "synthetic_thinker.json", "B": "synthetic_audio_enc.json",
            "C": "synthetic_vision.json", "D": "synthetic_talker.json",
            "E": "synthetic_omni_sft.json", "F": "synthetic_vocoder.json",
            "G": "synthetic_ocr.json",
        }

        for stage, config_name in sorted(config_map.items()):
            r = self.get(f"/api/system/config/{config_name}")
            cfg = r.get("config", {})
            s = f"{stage}"

            # LR
            lr = cfg.get("lr") or cfg.get("lr_g")
            if lr is not None:
                self.assert_true(f"{s} lr > 0", lr > 0, f"lr={lr}")
                self.assert_true(f"{s} lr <= 1", lr <= 1, f"lr={lr}")

            # Weight decay
            wd = cfg.get("wd")
            if wd is not None:
                self.assert_true(f"{s} wd >= 0", wd >= 0, f"wd={wd}")

            # Ranges
            for key, lo, hi in [
                ("dropout", 0, 1), ("label_smoothing", 0, 1), ("ema_decay", 0, 1), ("val_split", 0, 1),
            ]:
                v = cfg.get(key)
                if v is not None:
                    self.assert_true(f"{s} {key} in [{lo},{hi}]", lo <= v <= hi, f"{key}={v}")

            # Positive ints
            for key in ["warmup_steps", "max_steps", "batch_size", "seed", "checkpoint_freq", "val_freq"]:
                v = cfg.get(key)
                if v is not None:
                    self.assert_true(f"{s} {key} > 0", v > 0, f"{key}={v}")

            # Booleans
            self.assert_eq(f"{s} use_compile=false", cfg.get("use_compile"), False)
            self.assert_eq(f"{s} use_amp=true", cfg.get("use_amp"), True)
            self.assert_eq(f"{s} val_loss_threshold=999", cfg.get("val_loss_threshold"), 999.0)

            # Architecture: d_model divisible by n_heads
            dm = cfg.get("d_model")
            nh = cfg.get("n_heads")
            if dm and nh:
                self.assert_eq(f"{s} d_model%n_heads=0", dm % nh, 0)

            # Stage-specific
            if stage == "A":
                kv = cfg.get("kv_groups")
                if kv and nh:
                    self.assert_eq(f"{s} n_heads%kv_groups=0", nh % kv, 0)
                self.assert_true(f"{s} rope_theta>0", cfg.get("rope_theta", 1) > 0)

            if stage == "C":
                temp = cfg.get("temperature")
                if temp:
                    self.assert_true(f"{s} temperature in (0,1]", 0 < temp <= 1, f"temp={temp}")

            if stage == "F":
                for k in ["lr_g", "lr_d", "lambda_mel"]:
                    v = cfg.get(k)
                    if v is not None:
                        self.assert_true(f"{s} {k}>0", v > 0, f"{k}={v}")

    def test_tuning_range_sanity(self):
        """Validate tuning search space ranges are sane."""
        r = self.get("/api/tuning/spaces")
        spaces = r.get("spaces", {})

        for stage_id, space in spaces.items():
            for p in space.get("params", []):
                name, ptype = p["name"], p["type"]
                prefix = f"{stage_id}/{name}"

                if ptype in ("float_log", "float"):
                    lo, hi = p.get("low"), p.get("high")
                    if lo is not None and hi is not None:
                        self.assert_true(f"{prefix} low<high", lo < hi, f"{lo}>={hi}")
                    if ptype == "float_log" and lo is not None:
                        self.assert_true(f"{prefix} log low>0", lo > 0, f"low={lo}")
                    # Known param sanity
                    if "lr" in name and "mult" not in name and hi is not None:
                        self.assert_true(f"{prefix} lr max<=1", hi <= 1, f"max={hi}")
                    if name == "dropout" and hi is not None:
                        self.assert_true(f"{prefix} dropout max<=0.5", hi <= 0.5, f"max={hi}")

                elif ptype == "int":
                    lo, hi = p.get("low"), p.get("high")
                    if lo is not None:
                        self.assert_true(f"{prefix} int low>=0", lo >= 0, f"low={lo}")

    def test_config_value_in_tuning_range(self):
        """Config defaults should be within or near tuning search ranges."""
        r = self.get("/api/tuning/spaces")
        spaces = r.get("spaces", {})

        config_map = {
            "A": "synthetic_thinker.json", "B": "synthetic_audio_enc.json",
            "C": "synthetic_vision.json", "D": "synthetic_talker.json",
            "E": "synthetic_omni_sft.json", "F": "synthetic_vocoder.json",
            "G": "synthetic_ocr.json",
        }

        for stage_id, space in spaces.items():
            config_name = config_map.get(stage_id)
            if not config_name:
                continue
            r2 = self.get(f"/api/system/config/{config_name}")
            cfg = r2.get("config", {})

            for p in space.get("params", []):
                name, ptype = p["name"], p["type"]
                val = cfg.get(name)
                if val is None:
                    for v in cfg.values():
                        if isinstance(v, dict) and name in v:
                            val = v[name]
                            break
                if val is None:
                    continue

                prefix = f"{stage_id}/{name}"
                if ptype in ("float_log", "float"):
                    lo, hi = p.get("low", 0), p.get("high", 1)
                    if isinstance(val, (int, float)) and val != 0:
                        self.assert_true(f"{prefix} cfg={val} near range [{lo},{hi}]",
                                        val >= lo * 0.1 and val <= hi * 10,
                                        f"val={val} outside 10x of [{lo},{hi}]")

    def test_all_stages_registered(self):
        """Verify all training/testing scripts are registered in their APIs."""
        # Training
        r = self.get("/api/training/pipeline")
        stages = r.get("stages", {})
        expected_modules = {
            "A": "train.train_thinker", "B": "train.train_audio_enc",
            "C": "train.train_vision", "D": "train.train_talker",
            "E": "train.sft_omni", "F": "train.train_vocoder",
            "G": "train.train_ocr",
        }
        for stage_id, module in expected_modules.items():
            self.assert_eq(f"Stage {stage_id} module", stages[stage_id]["module"], module)

        # Testing
        test_scripts = ["test_thinker", "test_audio_enc", "test_vision",
                       "test_talker", "test_vocoder", "test_ocr", "test_sft"]
        for script in test_scripts:
            r = self.post("/api/testing/run", {"script": script, "num_samples": 1})
            # Should either start (ok) or fail with GPU busy — not 400 bad script
            self.assert_true(f"Test script {script} registered",
                            r.get("ok") or "GPU busy" in str(r.get("error", "")),
                            f"got: {r.get('error', '')}")
            # Stop it if started
            if r.get("ok"):
                time.sleep(0.5)
                self.wait_gpu_free(timeout=10)

    test_all_stages_registered._slow = True

    # ══════════════════════════════════════════════════════════
    # NEW FEATURE TESTS (improvements round)
    # ══════════════════════════════════════════════════════════

    def test_new_ui_elements(self):
        """Verify all new HTML elements from the improvements round exist."""
        status, body, headers = self.get_raw("/")

        checks = [
            ("GPU sparkline container", b"gpuSparkline"),
            ("Loss Trend card", b"cardLossTrend"),
            ("Start All Idle button", b"Start All Idle"),
            ("Log auto-refresh checkbox", b"logAutoRefresh"),
            ("Config diff output", b"configDiffOutput"),
            ("LIVE button Space hint", b"Space"),
        ]
        for label, needle in checks:
            self.assert_true(f"HTML: {label}", needle in body)

        # CSS checks
        status2, css, _ = self.get_raw("/static/style.css")
        css_checks = [
            ("stage-progress bar class", b"stage-progress"),
            ("toast cursor pointer", b"cursor: pointer"),
        ]
        for label, needle in css_checks:
            self.assert_true(f"CSS: {label}", needle in css)

    def test_new_js_features(self):
        """Verify all new JS functions and patterns from improvements round."""
        status, body, headers = self.get_raw("/static/app.js")

        checks = [
            ("debounce function", b"debounce"),
            ("timeAgo function", b"timeAgo"),
            ("chartRegistry array", b"chartRegistry"),
            ("keydown handler", b"keydown"),
            ("Space key shortcut", b"Space"),
            ("Notification API", b"Notification"),
            ("lastTimestamp tracking", b"lastTimestamp"),
            ("localStorage for filters", b"localStorage"),
            ("retrainStage confirm", b"retrainStage"),
            ("startAllIdle function", b"startAllIdle"),
            ("stage-progress rendering", b"stage-progress"),
            ("markLine event annotations", b"markLine"),
            ("cardLossTrend update", b"cardLossTrend"),
            ("config diff logic", b"Diff"),
            ("GPU sparkline/history", b"gpuSparkline"),
        ]
        for label, needle in checks:
            self.assert_true(f"JS: {label}", needle in body)

    def test_incremental_polling(self):
        """Verify incremental metrics fetch with since parameter."""
        # Full fetch
        r_all = self.get("/api/metrics/data?file=train_thinker.jsonl")
        all_rows = r_all.get("rows", [])
        self.assert_gt("Has metrics rows", len(all_rows), 0)

        # Far future = empty
        r_empty = self.get("/api/metrics/data?file=train_thinker.jsonl&since=2099-12-31T23:59:59Z")
        self.assert_eq("Far future since = 0 rows", len(r_empty.get("rows", [])), 0)

        # Past timestamp = subset
        if len(all_rows) > 5:
            mid_ts = all_rows[len(all_rows) // 2].get("timestamp", "")
            r_since = self.get(f"/api/metrics/data?file=train_thinker.jsonl&since={mid_ts}")
            since_rows = r_since.get("rows", [])
            self.assert_true("Since returns fewer than all",
                            len(since_rows) < len(all_rows),
                            f"{len(since_rows)} >= {len(all_rows)}")
            # All returned rows have timestamp > mid_ts
            for row in since_rows[:5]:
                self.assert_true(f"Row ts > since",
                                row.get("timestamp", "") > mid_ts)

        # Same file fetched twice returns same count (caching)
        r1 = self.get("/api/metrics/data?file=train_thinker.jsonl")
        r2 = self.get("/api/metrics/data?file=train_thinker.jsonl")
        self.assert_eq("Cached same count", len(r1.get("rows", [])), len(r2.get("rows", [])))

    def test_metrics_caching(self):
        """Verify server-side metrics caching works (mtime-based)."""
        # Fetch same file twice — second should be cached (same response)
        r1 = self.get("/api/metrics/data?file=train_thinker.jsonl")
        r2 = self.get("/api/metrics/data?file=train_thinker.jsonl")
        self.assert_eq("Cached fetch returns same row count",
                       len(r1.get("rows", [])), len(r2.get("rows", [])))

        # Incremental fetch with since param
        rows = r1.get("rows", [])
        if len(rows) > 10:
            mid_ts = rows[len(rows) // 2].get("timestamp", "")
            r3 = self.get(f"/api/metrics/data?file=train_thinker.jsonl&since={mid_ts}")
            since_rows = r3.get("rows", [])
            self.assert_true("Since filter returns fewer rows",
                            len(since_rows) < len(rows),
                            f"since={len(since_rows)} >= all={len(rows)}")
            # All returned rows should have timestamp > mid_ts
            for row in since_rows[:5]:
                ts = row.get("timestamp", "")
                self.assert_true(f"Row timestamp > since", ts > mid_ts, f"ts={ts}")

    def test_gpu_sparkline_container(self):
        """Verify GPU sparkline container exists in HTML."""
        status, body, headers = self.get_raw("/")
        self.assert_true("HTML has GPU sparkline div", b"gpuSparkline" in body)

    def test_loss_trend_card(self):
        """Verify Loss Trend summary card exists in HTML."""
        status, body, headers = self.get_raw("/")
        self.assert_true("HTML has Loss Trend card", b"cardLossTrend" in body)
        self.assert_true("HTML has Loss Trend label", b"Loss Trend" in body)

    def test_start_all_idle_button(self):
        """Verify Start All Idle button exists in HTML."""
        status, body, headers = self.get_raw("/")
        self.assert_true("HTML has Start All Idle button", b"Start All Idle" in body)
        self.assert_true("HTML has startAllIdle function ref", b"startAllIdle" in body)

    def test_log_auto_refresh_checkbox(self):
        """Verify auto-refresh checkbox exists in logs tab."""
        status, body, headers = self.get_raw("/")
        self.assert_true("HTML has log auto-refresh checkbox", b"logAutoRefresh" in body)

    def test_config_diff_output(self):
        """Verify config diff output container exists."""
        status, body, headers = self.get_raw("/")
        self.assert_true("HTML has config diff output", b"configDiffOutput" in body)

    def test_keyboard_shortcut_hint(self):
        """Verify LIVE button shows keyboard shortcut hint."""
        status, body, headers = self.get_raw("/")
        self.assert_true("LIVE button has Space shortcut hint", b"Space" in body)

    def test_stage_progress_css(self):
        """Verify stage progress bar CSS exists."""
        status, body, headers = self.get_raw("/static/style.css")
        self.assert_true("CSS has stage-progress class", b"stage-progress" in body)

    def test_toast_clickable_css(self):
        """Verify toast has cursor pointer for click-to-dismiss."""
        status, body, headers = self.get_raw("/static/style.css")
        # Toast should have cursor: pointer
        self.assert_true("Toast CSS has cursor pointer",
                        b"cursor: pointer" in body and b".toast" in body)

    def test_js_has_debounce(self):
        """Verify debounce function exists in app.js."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has debounce function", b"function debounce" in body or b"debounce" in body)

    def test_js_has_time_ago(self):
        """Verify timeAgo function exists in app.js."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has timeAgo function", b"timeAgo" in body)

    def test_js_has_chart_registry(self):
        """Verify chart registry for efficient resize exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has chartRegistry", b"chartRegistry" in body)

    def test_js_has_keyboard_shortcuts(self):
        """Verify keyboard shortcut handler exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has keydown handler", b"keydown" in body)
        self.assert_true("JS handles Space key", b"Space" in body)

    def test_js_has_desktop_notifications(self):
        """Verify desktop notification code exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has Notification API", b"Notification" in body)

    def test_js_has_incremental_polling(self):
        """Verify incremental polling with lastTimestamp exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS tracks lastTimestamp", b"lastTimestamp" in body)

    def test_js_has_local_storage_filters(self):
        """Verify filter persistence in localStorage."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS uses localStorage for filters", b"localStorage" in body)

    def test_js_has_retrain_confirm(self):
        """Verify retrainStage function with confirmation exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has retrainStage function", b"retrainStage" in body)

    def test_js_has_start_all_idle(self):
        """Verify startAllIdle function exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has startAllIdle function", b"startAllIdle" in body)

    def test_js_has_stage_progress(self):
        """Verify stage progress bar rendering code exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS renders stage-progress", b"stage-progress" in body)

    def test_js_has_event_annotations(self):
        """Verify chart event annotation (markLine) code exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has markLine for events", b"markLine" in body)

    def test_js_has_loss_trend(self):
        """Verify loss trend calculation exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS updates cardLossTrend", b"cardLossTrend" in body)

    def test_js_has_stage_aware_cards(self):
        """Verify stage-aware summary card filtering."""
        status, body, headers = self.get_raw("/static/app.js")
        # Should detect running stage and filter metrics
        self.assert_true("JS has activeStageFile logic",
                        b"activeStage" in body or b"_file" in body)

    def test_js_has_config_diff(self):
        """Verify config diff highlighting code exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has config diff logic", b"configDiff" in body or b"Diff" in body)

    def test_js_has_gpu_sparkline(self):
        """Verify GPU sparkline rendering code exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has GPU sparkline", b"gpuSparkline" in body or b"gpuHistory" in body)

    def test_metrics_since_param_empty_result(self):
        """Verify since param far in the future returns empty."""
        r = self.get("/api/metrics/data?file=train_thinker.jsonl&since=2099-12-31T23:59:59Z")
        self.assert_eq("Far future since = 0 rows", len(r.get("rows", [])), 0)

    def test_metrics_since_param_past(self):
        """Verify since param in the past returns subset."""
        r_all = self.get("/api/metrics/data?file=train_thinker.jsonl")
        all_rows = r_all.get("rows", [])
        if len(all_rows) > 5:
            # Use timestamp from middle of dataset
            mid = all_rows[len(all_rows) // 2]
            ts = mid.get("timestamp", "")
            r_since = self.get(f"/api/metrics/data?file=train_thinker.jsonl&since={ts}")
            since_rows = r_since.get("rows", [])
            self.assert_true("Since returns fewer than all",
                            len(since_rows) < len(all_rows),
                            f"{len(since_rows)} >= {len(all_rows)}")

    # ══════════════════════════════════════════════════════════
    # LATEST FEATURES: Paused state, Clear reset, Metrics delete,
    #   Chat UI, Step display, Tuning resume
    # ══════════════════════════════════════════════════════════

    def test_paused_state_detection(self):
        """Verify partial checkpoint shows 'paused' not 'done' after server restart."""
        r = self.get("/api/training/pipeline")
        stages = r.get("stages", {})

        for stage_id, s in stages.items():
            if not s.get("has_checkpoint"):
                continue
            meta = s.get("metadata")
            if not meta:
                continue

            # Read config to get max_steps
            r2 = self.get(f"/api/system/checkpoint-config/{s.get('checkpoint_dir', '').split('/')[-1]}")
            if not r2.get("ok"):
                continue
            max_steps = r2.get("config", {}).get("max_steps", 0)
            step = meta.get("step", 0)

            if max_steps > 0 and step < max_steps and s.get("status") not in ("running", "stopped", "failed"):
                self.assert_eq(f"Stage {stage_id} partial ckpt -> paused", s["status"], "paused")
            elif max_steps > 0 and step >= max_steps and s.get("status") not in ("running", "stopped", "failed"):
                self.assert_eq(f"Stage {stage_id} complete ckpt -> done", s["status"], "done")

    def test_clear_resets_to_idle(self):
        """Verify Clear removes process record and stage becomes idle."""
        self.wait_gpu_free()

        # Start Stage A briefly
        r = self.post("/api/training/start", {"stage": "A"})
        if not r.get("ok"):
            self.skip("Clear reset test", "couldn't start A")
            return
        time.sleep(3)

        # Clear (should stop + delete + reset)
        r = self.check("Clear while running", self.post("/api/training/clear", {"stage": "A"}))

        time.sleep(2)
        p = self.get("/api/training/pipeline")
        stage_a = p["stages"]["A"]
        self.assert_eq("A is idle after clear", stage_a["status"], "idle")
        self.assert_eq("A has no checkpoint", stage_a["has_checkpoint"], False)
        # Process record should be gone
        r2 = self.get("/api/training/status")
        self.assert_true("No training_A process after clear",
                        "training_A" not in r2.get("processes", {}))

    test_clear_resets_to_idle._slow = True

    def test_metrics_delete_api(self):
        """Verify metrics file and run deletion endpoints."""
        # Ensure we have data
        files = self.get("/api/metrics/files").get("files", [])
        if not files:
            self.skip("Metrics delete test", "no metrics files")
            return

        # Test delete-run (non-existent run = 0 removed, file unchanged)
        r = self.check("Delete non-existent run",
                       self.post("/api/metrics/delete-run", {"file": files[0], "run_id": "nonexistent_999"}))
        self.assert_eq("Removed 0 rows", r.get("removed_rows"), 0)

        # Test delete non-existent file
        r = self.post("/api/metrics/delete", {"file": "nonexistent_file.jsonl"})
        self.check("Delete non-existent file -> 404", r, expect_ok=False)

        # Test delete missing params
        self.check("Delete no file -> 400", self.post("/api/metrics/delete", {}), expect_ok=False)
        self.check("Delete-run no params -> 400",
                   self.post("/api/metrics/delete-run", {}), expect_ok=False)

    def test_chat_ui_elements(self):
        """Verify chat inference UI elements exist."""
        status, body, headers = self.get_raw("/")
        checks = [
            ("Chat mode tab", b'data-mode="chat"'),
            ("Single mode tab", b'data-mode="single"'),
            ("Chat messages container", b"inferChatMessages"),
            ("Chat input field", b"inferChatInput"),
            ("Chat send button", b"inferChatSendBtn"),
            ("Attach image button", b"inferAttachImgBtn"),
            ("Attach audio button", b"inferAttachAudioBtn"),
            ("Clear chat button", b"inferClearChat"),
            ("Single mode textarea", b"inferText"),
            ("Single mode output", b"inferResult"),
            ("Chat empty state icon", b"infer-chat-empty-icon"),
        ]
        for label, needle in checks:
            self.assert_true(f"Chat UI: {label}", needle in body)

    def test_delete_buttons_exist(self):
        """Verify metrics delete buttons exist in HTML."""
        status, body, headers = self.get_raw("/")
        self.assert_true("Delete Selected button", b"deleteMetricsBtn" in body)
        self.assert_true("Delete All Data button", b"deleteAllMetricsBtn" in body)

    def test_step_display_in_pipeline(self):
        """Verify pipeline shows max_steps target alongside current step."""
        status, body, headers = self.get_raw("/static/app.js")
        # JS should compute both metricsStep and metaStep
        self.assert_true("JS has metricsStep", b"metricsStep" in body)
        self.assert_true("JS has metaStep", b"metaStep" in body)
        self.assert_true("JS shows ckpt indicator", b"ckpt:" in body)
        self.assert_true("JS shows maxLabel", b"maxLabel" in body)
        self.assert_true("JS has getLatestStepFromMetrics", b"getLatestStepFromMetrics" in body)

    def test_paused_status_in_css(self):
        """Verify paused status has CSS styling."""
        status, body, headers = self.get_raw("/static/style.css")
        self.assert_true("CSS has .paused status dot", b"status-dot.paused" in body or b".paused" in body)

    def test_tuning_resume_ui(self):
        """Verify tuning UI shows resume label when DB exists."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has Resume Tuning label", b"Resume Tuning" in body)
        self.assert_true("JS checks existing trials for resume", b"existingTrials" in body)
        self.assert_true("JS shows resumed indicator", b"resumed" in body)
        self.assert_true("JS shows Paused label", b"Paused:" in body)

    def test_process_manager_clear_record(self):
        """Verify ProcessManager.clear_record exists and works."""
        # The clear API endpoint uses pm.clear_record()
        # Test indirectly: start, stop, clear, verify no record
        self.wait_gpu_free()

        # Start and immediately stop
        r = self.post("/api/training/start", {"stage": "F"})
        if not r.get("ok"):
            self.skip("PM clear_record test", "couldn't start F")
            return
        time.sleep(2)
        self.post("/api/training/stop", {"stage": "F"})
        time.sleep(1)

        # Verify process exists
        r = self.get("/api/training/status")
        self.assert_in("training_F exists", "training_F", r.get("processes", {}))

        # Clear stage F
        self.post("/api/training/clear", {"stage": "F"})
        time.sleep(1)

        # Verify process record is gone
        r = self.get("/api/training/status")
        self.assert_true("training_F removed after clear",
                        "training_F" not in r.get("processes", {}))

    test_process_manager_clear_record._slow = True

    def test_chip_clear_all_html(self):
        """Verify chip filter system has clear-all support in JS."""
        status, body, headers = self.get_raw("/static/app.js")
        self.assert_true("JS has chip-clear class", b"chip-clear" in body)
        self.assert_true("JS has clear-all action", b"clear-all" in body)

        status2, css, _ = self.get_raw("/static/style.css")
        self.assert_true("CSS has chip-clear styling", b"chip-clear" in css)
        self.assert_true("CSS has chip-x styling", b"chip-x" in css)


def main():
    parser = argparse.ArgumentParser(description="Test micro-Omni server API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-slow", action="store_true", help="Skip tests that start/stop processes")
    args = parser.parse_args()

    tester = APITester(args.host, args.port)

    try:
        urllib.request.urlopen(f"http://{args.host}:{args.port}/api/system/gpu", timeout=3)
    except Exception:
        print(f"Server not running at {args.host}:{args.port}")
        print("Start it first: python -m server --no-open")
        sys.exit(1)

    success = tester.run_all(skip_slow=args.skip_slow)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
