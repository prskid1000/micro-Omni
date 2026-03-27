"""Comprehensive API test suite for the micro-Omni unified server.

Usage:
    # Start server first: python -m server --no-open
    python server/test_api.py
    python server/test_api.py --host 127.0.0.1 --port 8000
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
        self.errors: list[str] = []

    def get(self, path: str) -> dict:
        try:
            r = urllib.request.urlopen(f"{self.base}{path}", timeout=10)
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
            r = urllib.request.urlopen(req, timeout=10)
            return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def check(self, label: str, result: dict, expect_ok: bool = True) -> dict:
        ok = result.get("ok", False)
        passed = ok == expect_ok
        status = "PASS" if passed else "FAIL"

        detail = ""
        if not ok and expect_ok:
            detail = f' -> {result.get("error", "")}'
        elif ok and not expect_ok:
            detail = " -> expected failure but succeeded"

        print(f"  [{status}] {label}{detail}")

        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"{label}{detail}")

        return result

    def assert_eq(self, label: str, actual, expected):
        passed = actual == expected
        status = "PASS" if passed else "FAIL"
        detail = "" if passed else f" -> got {actual!r}, expected {expected!r}"
        print(f"  [{status}] {label}{detail}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"{label}{detail}")

    def wait_gpu_free(self, timeout: int = 120):
        """Wait until no GPU process is running."""
        for i in range(timeout // 2):
            r = self.get("/api/training/pipeline")
            stages = r.get("stages", {})
            any_running = any(s.get("status") == "running" for s in stages.values())
            r2 = self.get("/api/testing/status")
            test_running = any(p.get("status") == "running" for p in r2.get("processes", {}).values())
            if not any_running and not test_running:
                return True
            time.sleep(2)
        return False

    def run_all(self):
        print("=" * 64)
        print("  micro-Omni Server API Test Suite")
        print("=" * 64)

        self.test_static_files()
        self.test_metrics_api()
        self.test_system_api()
        self.test_config_roundtrip()
        self.test_training_lifecycle()
        self.test_gpu_lock()
        self.test_testing_api()
        self.test_inference_api()
        self.test_tuning_api()
        self.test_export_api()

        print()
        print("=" * 64)
        total = self.passed + self.failed
        print(f"  Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print()
            print("  Failures:")
            for e in self.errors:
                print(f"    - {e}")
        print("=" * 64)
        return self.failed == 0

    # ── Static Files ─────────────────────────────────────────

    def test_static_files(self):
        print()
        print("--- Static Files ---")
        for path in ["/", "/static/style.css", "/static/app.js"]:
            try:
                r = urllib.request.urlopen(f"{self.base}{path}", timeout=5)
                size = len(r.read())
                self.check(f"GET {path} ({size} bytes)", {"ok": True})
            except Exception as e:
                self.check(f"GET {path}", {"ok": False, "error": str(e)})

    # ── Metrics API ──────────────────────────────────────────

    def test_metrics_api(self):
        print()
        print("--- Metrics API ---")
        self.check("GET /api/metrics/files", self.get("/api/metrics/files"))
        self.check("GET /api/metrics/summary", self.get("/api/metrics/summary"))

        r = self.check("GET /api/metrics/data?file=__all__", self.get("/api/metrics/data?file=__all__"))

        # Test incremental fetch with since param
        self.check("GET /api/metrics/data?file=train_thinker.jsonl&since=2099-01-01",
                    self.get("/api/metrics/data?file=train_thinker.jsonl&since=2099-01-01"))

        # Bad file name
        self.check("GET /api/metrics/data (no file)", self.get("/api/metrics/data"), expect_ok=False)

    # ── System API ───────────────────────────────────────────

    def test_system_api(self):
        print()
        print("--- System API ---")
        r = self.check("GET /api/system/gpu", self.get("/api/system/gpu"))
        if r.get("gpu"):
            print(f"       GPU: {r['gpu'].get('name')}, {r['gpu'].get('memory_used_mb')}MB/{r['gpu'].get('memory_total_mb')}MB")

        r = self.check("GET /api/system/checkpoints", self.get("/api/system/checkpoints"))
        print(f"       Checkpoints: {len(r.get('checkpoints', []))}")

        r = self.check("GET /api/system/configs", self.get("/api/system/configs"))
        print(f"       Configs: {len(r.get('configs', []))}")

        self.check("GET /api/system/config/synthetic_thinker.json",
                    self.get("/api/system/config/synthetic_thinker.json"))

        # Nonexistent config
        self.check("GET /api/system/config/nonexistent.json",
                    self.get("/api/system/config/nonexistent.json"), expect_ok=False)

    # ── Config Roundtrip ─────────────────────────────────────

    def test_config_roundtrip(self):
        print()
        print("--- Config Save/Read Roundtrip ---")
        r = self.get("/api/system/config/synthetic_thinker.json")
        original_cfg = r.get("config", {})
        original_lr = original_cfg.get("lr")

        # Modify and save
        modified_cfg = dict(original_cfg)
        modified_cfg["lr"] = 0.99999
        self.check("POST save modified config",
                    self.post("/api/system/config/synthetic_thinker.json", {"config": modified_cfg}))

        # Read back and verify
        r = self.get("/api/system/config/synthetic_thinker.json")
        self.assert_eq("Config lr matches saved value", r.get("config", {}).get("lr"), 0.99999)

        # Restore original
        self.check("POST restore original config",
                    self.post("/api/system/config/synthetic_thinker.json", {"config": original_cfg}))
        r = self.get("/api/system/config/synthetic_thinker.json")
        self.assert_eq("Config lr restored", r.get("config", {}).get("lr"), original_lr)

    # ── Training Lifecycle ───────────────────────────────────

    def test_training_lifecycle(self):
        print()
        print("--- Training Lifecycle ---")

        # Ensure GPU free
        self.wait_gpu_free()

        # Start Stage A
        r = self.check("POST start Stage A", self.post("/api/training/start", {"stage": "A"}))
        pid = r.get("pid")
        print(f"       PID: {pid}")
        time.sleep(3)

        # Verify running via pipeline
        r = self.get("/api/training/pipeline")
        self.assert_eq("Stage A status = running", r.get("stages", {}).get("A", {}).get("status"), "running")

        # Get logs
        r = self.check("GET /api/training/logs/A", self.get("/api/training/logs/A"))
        print(f"       Log lines: {len(r.get('lines', []))}")

        # Can't clear while running
        self.check("POST clear A while running (expect 409)",
                    self.post("/api/training/clear", {"stage": "A"}), expect_ok=False)

        # Stop
        r = self.check("POST stop Stage A", self.post("/api/training/stop", {"stage": "A"}))
        self.assert_eq("Stopped = True", r.get("stopped"), True)
        time.sleep(2)

        # Verify stopped
        r = self.get("/api/training/pipeline")
        status_a = r.get("stages", {}).get("A", {}).get("status")
        # Status should be "stopped" or "done" (if it saved a checkpoint)
        ok = status_a in ("stopped", "done", "idle")
        self.check(f"Stage A status after stop = {status_a}", {"ok": ok})

        # Resume (start again — training scripts auto-resume from checkpoint)
        r = self.check("POST resume Stage A", self.post("/api/training/start", {"stage": "A"}))
        resume_pid = r.get("pid")
        print(f"       Resume PID: {resume_pid}")
        time.sleep(3)

        # Verify running
        r = self.get("/api/training/pipeline")
        self.assert_eq("Stage A resumed = running",
                       r.get("stages", {}).get("A", {}).get("status"), "running")

        # Stop again
        self.check("POST stop Stage A again", self.post("/api/training/stop", {"stage": "A"}))
        time.sleep(2)

        # Clear checkpoint
        r = self.check("POST clear Stage A checkpoint", self.post("/api/training/clear", {"stage": "A"}))
        print(f"       Cleared: {r.get('count', 0)} files")

        # Verify idle
        r = self.get("/api/training/pipeline")
        stage_a = r.get("stages", {}).get("A", {})
        self.assert_eq("Stage A idle after clear", stage_a.get("has_checkpoint"), False)

        # Verify D blocked by A
        stage_d = r.get("stages", {}).get("D", {})
        self.assert_eq("Stage D blocked by A", "A" in stage_d.get("blocked_by", []), True)

    # ── GPU Lock ─────────────────────────────────────────────

    def test_gpu_lock(self):
        print()
        print("--- GPU Lock (Single-GPU Enforcement) ---")

        self.wait_gpu_free()

        # Start Stage B
        r = self.check("POST start Stage B", self.post("/api/training/start", {"stage": "B"}))
        time.sleep(2)

        # Try all other GPU operations — all should 409
        self.check("POST start Stage C (GPU busy)",
                    self.post("/api/training/start", {"stage": "C"}), expect_ok=False)
        self.check("POST start test (GPU busy)",
                    self.post("/api/testing/run", {"script": "test_thinker"}), expect_ok=False)
        self.check("POST start tuning (GPU busy)",
                    self.post("/api/tuning/start", {"stage": "A", "n_trials": 1}), expect_ok=False)
        self.check("POST start export (GPU busy)",
                    self.post("/api/export/run", {}), expect_ok=False)
        self.check("POST inference chat (GPU busy)",
                    self.post("/api/inference/chat", {"text": "hello"}), expect_ok=False)

        # Stop B
        self.check("POST stop Stage B", self.post("/api/training/stop", {"stage": "B"}))
        time.sleep(2)

        # GPU should be free now
        r = self.get("/api/training/pipeline")
        b_running = r.get("stages", {}).get("B", {}).get("status") == "running"
        self.assert_eq("GPU free after stopping B", b_running, False)

    # ── Testing API ──────────────────────────────────────────

    def test_testing_api(self):
        print()
        print("--- Testing API ---")

        self.wait_gpu_free()

        self.check("GET /api/testing/status", self.get("/api/testing/status"))
        self.check("GET /api/testing/results/test_thinker",
                    self.get("/api/testing/results/test_thinker"))

        # Bad script name
        self.check("POST run nonexistent test",
                    self.post("/api/testing/run", {"script": "test_nonexistent"}), expect_ok=False)

    # ── Inference API ────────────────────────────────────────

    def test_inference_api(self):
        print()
        print("--- Inference API ---")

        # Unload (always safe)
        self.check("POST /api/inference/unload", self.post("/api/inference/unload"))

        # Missing text
        self.check("POST /api/inference/chat (no input)",
                    self.post("/api/inference/chat", {}), expect_ok=False)

        # Missing model_dir for standalone
        self.check("POST /api/inference/standalone (no text)",
                    self.post("/api/inference/standalone", {}), expect_ok=False)

        # Bad endpoint
        self.check("POST /api/inference/nonexistent",
                    self.post("/api/inference/nonexistent", {}), expect_ok=False)

    # ── Tuning API ───────────────────────────────────────────

    def test_tuning_api(self):
        print()
        print("--- Tuning API ---")

        self.check("GET /api/tuning/spaces", self.get("/api/tuning/spaces"))

        r = self.check("GET /api/tuning/status", self.get("/api/tuning/status"))

        # Results for stage with no DB
        self.check("GET /api/tuning/results/A (no DB)", self.get("/api/tuning/results/A"))

        # Bad stage
        self.check("POST /api/tuning/start (bad stage)",
                    self.post("/api/tuning/start", {"stage": "Z"}), expect_ok=False)

        # Clear already empty
        self.check("POST /api/tuning/clear A (already empty)",
                    self.post("/api/tuning/clear", {"stage": "A"}))

        # Apply with no results
        self.check("POST /api/tuning/apply A (no results)",
                    self.post("/api/tuning/apply", {"stage": "A"}), expect_ok=False)

    # ── Export API ───────────────────────────────────────────

    def test_export_api(self):
        print()
        print("--- Export API ---")
        self.check("GET /api/export/status", self.get("/api/export/status"))


def main():
    parser = argparse.ArgumentParser(description="Test micro-Omni server API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tester = APITester(args.host, args.port)

    # Check server is running
    try:
        urllib.request.urlopen(f"http://{args.host}:{args.port}/api/system/gpu", timeout=3)
    except Exception:
        print(f"Server not running at {args.host}:{args.port}")
        print("Start it first: python -m server --no-open")
        sys.exit(1)

    success = tester.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
