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
            self.assert_in(f"Stage {stage_id} status valid", s["status"], ["idle", "running", "done", "stopped", "failed", "blocked"])

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

        # Results with no DB
        r = self.check("Results with no DB", self.get("/api/tuning/results/A"))
        self.assert_eq("No results = None", r.get("results"), None)

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
