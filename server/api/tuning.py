"""HP Tuning API — define search spaces, launch/monitor Optuna studies per stage."""

from __future__ import annotations

import json
import os
import threading
from typing import Any

# Lock to prevent DB reads during clear
_db_lock = threading.Lock()

# ── Search space definitions per stage ────────────────────────
# Each entry: (param_name, type, args)
# Types: "float_log", "float", "int", "categorical"

COMMON_TRAINING_SPACE = [
    # Optimization
    ("lr", "float_log", (1e-4, 5e-3)),
    ("wd", "float_log", (1e-6, 0.1)),
    ("warmup_steps", "int", (10, 300)),
    ("batch_size", "categorical", ([8, 16, 32, 64],)),
    ("gradient_accumulation_steps", "categorical", ([1, 2, 4],)),
    ("max_grad_norm", "categorical", ([0.5, 1.0, 2.0, 5.0],)),
    # Regularization
    ("dropout", "float", (0.0, 0.2)),
    ("label_smoothing", "float", (0.0, 0.15)),
    ("ema_decay", "float", (0.99, 0.9999)),
    # Convergence control
    ("lr_spike_multiplier", "float", (2.0, 10.0)),
    ("lr_spike_duration", "int", (20, 100)),
    ("early_stopping_patience", "int", (3, 15)),
]

STAGE_SPECIFIC_SPACE: dict[str, list[tuple]] = {
    "A": [
        # Thinker LLM — core text model
        *COMMON_TRAINING_SPACE,
        ("kv_groups", "categorical", ([1, 2, 4],)),
        ("use_swiglu", "categorical", ([True, False],)),
        ("use_gqa", "categorical", ([True, False],)),
        ("rope_theta", "categorical", ([5000, 10000, 50000, 100000],)),
        ("use_moe", "categorical", ([True, False],)),
        ("use_flash", "categorical", ([True, False],)),
        ("use_spiking", "categorical", ([True, False],)),
        ("use_ltc", "categorical", ([True, False],)),
        ("shuffle_buffer_size", "categorical", ([5000, 10000, 20000],)),
        # Found in script cfg.get() but not in config — hidden tunable params
        ("use_mtp", "categorical", ([True, False],)),
        ("window_size", "categorical", ([None, 32, 64, 128],)),
        ("rope_scaling_factor", "categorical", ([1.0, 2.0, 4.0],)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
    "B": [
        # Audio Encoder — mel → embeddings
        *COMMON_TRAINING_SPACE,
        ("use_attention_pooling", "categorical", ([True, False],)),
        ("use_augmentation", "categorical", ([True, False],)),
        ("downsample_time", "categorical", ([4, 8, 16],)),
        ("target_hz", "categorical", ([6.25, 12.5, 25.0],)),
        ("use_flash", "categorical", ([True, False],)),
        ("use_spiking", "categorical", ([True, False],)),
        ("use_ltc", "categorical", ([True, False],)),
        ("shuffle_buffer_size", "categorical", ([5000, 10000, 20000],)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
    "C": [
        # Vision Encoder — CLIP contrastive
        *COMMON_TRAINING_SPACE,
        ("temperature", "float", (0.01, 0.2)),
        ("use_augmentation", "categorical", ([True, False],)),
        ("embed_dim", "categorical", ([64, 128, 256],)),
        ("use_thinker_for_text", "categorical", ([True, False],)),
        ("shuffle_buffer_size", "categorical", ([5000, 10000, 20000],)),
        # Vision script reads thinker sub-params
        ("use_gqa", "categorical", ([True, False],)),
        ("use_swiglu", "categorical", ([True, False],)),
        ("use_moe", "categorical", ([True, False],)),
        ("use_spiking", "categorical", ([True, False],)),
        ("use_ltc", "categorical", ([True, False],)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
    "D": [
        # Talker TTS — RVQ speech codes
        *COMMON_TRAINING_SPACE,
        ("use_gqa", "categorical", ([True, False],)),
        ("use_swiglu", "categorical", ([True, False],)),
        ("use_flash", "categorical", ([True, False],)),
        ("use_spiking", "categorical", ([True, False],)),
        ("use_ltc", "categorical", ([True, False],)),
        ("rvq_ema_decay", "float", (0.9, 0.999)),
        ("rvq_gumbel_temp", "float", (0.1, 2.0)),
        ("rvq_reset_threshold", "float", (0.5, 5.0)),
        ("codebooks", "categorical", ([1, 2, 4],)),
        ("codebook_size", "categorical", ([64, 128, 256],)),
        ("rope_theta", "categorical", ([5000, 10000, 50000],)),
        ("frame_rate", "categorical", ([6.25, 12.5, 25.0],)),
        ("shuffle_buffer_size", "categorical", ([5000, 10000, 20000],)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
    "E": [
        # Multimodal SFT — frozen encoders, train projectors + thinker
        *COMMON_TRAINING_SPACE,
        ("proj_lr_mult", "float", (1.0, 20.0)),
        ("val_batch_size", "categorical", ([2, 4, 8],)),
        ("use_flash", "categorical", ([True, False],)),
        ("shuffle_buffer_size", "categorical", ([5000, 10000, 20000],)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
    "F": [
        # Vocoder — HiFi-GAN (different LR structure, no common lr)
        ("lr_g", "float_log", (1e-5, 1e-3)),
        ("lr_d", "float_log", (1e-5, 1e-3)),
        ("wd", "float_log", (1e-6, 0.1)),
        ("warmup_steps", "int", (10, 200)),
        ("batch_size", "categorical", ([1, 2, 4],)),
        ("gradient_accumulation_steps", "categorical", ([1, 2, 4, 8],)),
        ("max_grad_norm", "categorical", ([0.5, 1.0, 2.0, 5.0],)),
        ("ema_decay", "float", (0.99, 0.9999)),
        ("lambda_mel", "float", (10.0, 100.0)),
        ("lambda_fm", "float", (1.0, 20.0)),
        ("lambda_adv", "float", (0.5, 5.0)),
        ("discriminator_update_interval", "categorical", ([1, 2, 3, 5],)),
        ("discriminator_lr_warmup_steps", "int", (50, 500)),
        ("mel_weight_decay_start", "int", (500, 5000)),
        ("mel_weight_decay_rate", "float", (0.0, 0.01)),
        ("mpd_periods", "categorical", ([[2, 3, 5, 7, 11], [2, 3, 5, 7], [3, 5, 7, 11, 13]],)),
        ("msd_num_scales", "categorical", ([2, 3, 4],)),
        ("shuffle_buffer_size", "categorical", ([500, 1000, 2000],)),
        ("lr_spike_multiplier", "float", (2.0, 10.0)),
        ("lr_spike_duration", "int", (20, 100)),
        ("early_stopping_patience", "int", (3, 15)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
    "G": [
        # OCR — encoder-decoder with separate stacks
        *COMMON_TRAINING_SPACE,
        ("use_gqa", "categorical", ([True, False],)),
        ("use_swiglu", "categorical", ([True, False],)),
        ("use_flash", "categorical", ([True, False],)),
        ("use_spiking", "categorical", ([True, False],)),
        ("use_ltc", "categorical", ([True, False],)),
        ("shuffle_buffer_size", "categorical", ([5000, 10000, 20000],)),
        ("early_stopping_min_delta", "float", (0.00001, 0.001)),
    ],
}

# Stage to training module + config mapping
from server.api.training import STAGE_MAP


def _get_search_space(stage: str) -> list[dict[str, Any]]:
    """Return search space as list of dicts for the UI."""
    raw = STAGE_SPECIFIC_SPACE.get(stage, COMMON_TRAINING_SPACE)
    result = []
    for item in raw:
        name, kind, args = item
        entry: dict[str, Any] = {"name": name, "type": kind}
        if kind == "float_log":
            entry["low"], entry["high"] = args
        elif kind == "float":
            entry["low"], entry["high"] = args
        elif kind == "int":
            entry["low"], entry["high"] = args
        elif kind == "categorical":
            entry["choices"] = args[0]
        result.append(entry)
    return result


def _read_study_results(db_path: str) -> dict[str, Any] | None:
    """Read Optuna study results directly from SQLite (no lingering connections)."""
    if not os.path.exists(db_path):
        return None
    import sqlite3
    from datetime import datetime

    with _db_lock:
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT study_id, study_name FROM studies LIMIT 1")
            study_row = cur.fetchone()
            if not study_row:
                return None
            study_name = study_row["study_name"]
            study_id = study_row["study_id"]

            # Direction is in a separate table in Optuna 4.x
            cur.execute("SELECT direction FROM study_directions WHERE study_id = ? LIMIT 1", (study_id,))
            dir_row = cur.fetchone()
            direction = dir_row["direction"] if dir_row else "MINIMIZE"

            cur.execute("""
                SELECT trial_id, number, state, datetime_start, datetime_complete
                FROM trials WHERE study_id = ? ORDER BY number
            """, (study_id,))
            trial_rows = cur.fetchall()

            # Optuna 4.x uses VARCHAR state, older uses INT
            STATE_INT_MAP = {0: "RUNNING", 1: "COMPLETE", 2: "PRUNED", 3: "FAIL", 4: "WAITING"}
            def _parse_state(raw):
                if isinstance(raw, int):
                    return STATE_INT_MAP.get(raw, "UNKNOWN")
                return str(raw).upper() if raw else "UNKNOWN"

            trials = []
            for tr in trial_rows:
                trial_id = tr["trial_id"]

                cur.execute("SELECT value FROM trial_values WHERE trial_id = ? AND objective = 0", (trial_id,))
                val_row = cur.fetchone()
                value = val_row["value"] if val_row else None

                cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?", (trial_id,))
                params = {}
                for pr in cur.fetchall():
                    name = pr["param_name"]
                    raw_val = pr["param_value"]  # stored as FLOAT
                    dist_json = pr["distribution_json"]
                    # Decode based on distribution type
                    try:
                        dist = json.loads(dist_json) if dist_json else {}
                        dist_name = dist.get("name", "")
                        if "Int" in dist_name:
                            params[name] = int(raw_val)
                        elif "Categorical" in dist_name:
                            choices = dist.get("attributes", {}).get("choices", [])
                            idx = int(raw_val)
                            params[name] = choices[idx] if idx < len(choices) else raw_val
                        else:
                            params[name] = raw_val
                    except Exception:
                        params[name] = raw_val

                duration = None
                if tr["datetime_start"] and tr["datetime_complete"]:
                    try:
                        t0 = datetime.fromisoformat(tr["datetime_start"])
                        t1 = datetime.fromisoformat(tr["datetime_complete"])
                        duration = (t1 - t0).total_seconds()
                    except Exception:
                        pass

                trials.append({
                    "number": tr["number"],
                    "value": value,
                    "params": params,
                    "state": _parse_state(tr["state"]),
                    "duration_seconds": duration,
                })

            complete = [t for t in trials if t["state"] == "COMPLETE" and t["value"] is not None]
            best = None
            if complete:
                best_t = min(complete, key=lambda t: t["value"]) if direction == "MINIMIZE" else max(complete, key=lambda t: t["value"])
                best = {"number": best_t["number"], "value": best_t["value"], "params": best_t["params"]}

            return {
                "study_name": study_name,
                "direction": direction,
                "n_trials": len(trials),
                "best_trial": best,
                "trials": trials,
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()


def handle_get(handler: Any, path: str, query: dict[str, list[str]]) -> None:
    if path == "/api/tuning/spaces":
        spaces = {}
        for stage_id in STAGE_MAP:
            spaces[stage_id] = {
                "name": STAGE_MAP[stage_id]["name"],
                "params": _get_search_space(stage_id),
            }
        handler.send_json({"ok": True, "spaces": spaces})
        return

    if path.startswith("/api/tuning/results/"):
        stage = path.split("/")[-1].upper()
        db_path = f"logs/hp_tuning_{stage}.db"
        results = _read_study_results(db_path)

        # Load saved tuning config if exists
        config_path = f"logs/hp_tuning_{stage}_config.json"
        tune_config = None
        if os.path.exists(config_path):
            try:
                tune_config = json.loads(open(config_path, "r", encoding="utf-8").read())
            except Exception:
                pass

        handler.send_json({"ok": True, "stage": stage, "results": results, "tune_config": tune_config})
        return

    if path == "/api/tuning/status":
        from server.app import get_process_manager
        pm = get_process_manager()
        processes = pm.get_all(category="tuning")
        handler.send_json({"ok": True, "processes": processes})
        return

    handler.send_error_json(404, f"Unknown tuning endpoint: {path}")


def handle_post(handler: Any, path: str, body: dict[str, Any]) -> None:
    from server.app import get_process_manager
    pm = get_process_manager()

    if path == "/api/tuning/start":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return

        n_trials = body.get("n_trials", 30)
        max_steps = body.get("max_steps", 500)
        params = body.get("params")  # optional: subset of params to tune

        # Save tuning config for UI persistence across restarts
        config_path = f"logs/hp_tuning_{stage}_config.json"
        os.makedirs("logs", exist_ok=True)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "stage": stage,
                    "n_trials": n_trials,
                    "max_steps": max_steps,
                    "params": params,
                    "config": STAGE_MAP[stage]["config"],
                    "started_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception:
            pass

        extra_args = [
            "--stage", stage,
            "--n_trials", str(n_trials),
            "--max_steps", str(max_steps),
        ]
        if params:
            extra_args += ["--params", json.dumps(params)]

        try:
            mp = pm.start(
                category="tuning",
                stage=f"tune_{stage}",
                module="train.tune",
                extra_args=extra_args,
            )
            handler.send_json({
                "ok": True,
                "pid": mp.pid,
                "stage": stage,
                "n_trials": n_trials,
                "max_steps": max_steps,
            })
        except RuntimeError as e:
            handler.send_error_json(409, str(e))
        return

    if path == "/api/tuning/stop":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return
        key = f"tuning_tune_{stage}"
        stopped = pm.stop(key)
        handler.send_json({"ok": True, "stage": stage, "stopped": stopped})
        return

    if path == "/api/tuning/clear":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return

        removed = []

        # 1. Delete Optuna DB + WAL/journal files (acquire lock to ensure no readers)
        with _db_lock:
            db_path = f"logs/hp_tuning_{stage}.db"
            for suffix in ["", "-wal", "-shm", "-journal"]:
                p = db_path + suffix
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        removed.append(p)
                    except Exception as e:
                        handler.send_error_json(500, f"Failed to delete {p}: {e}")
                        return

        # 2. Delete tuning checkpoint dirs (checkpoints/tune_<stage>/trial_*)
        import shutil
        tune_dir = os.path.join("checkpoints", f"tune_{stage}")
        if os.path.exists(tune_dir):
            try:
                shutil.rmtree(tune_dir)
                removed.append(tune_dir)
            except Exception as e:
                handler.send_error_json(500, f"Failed to delete {tune_dir}: {e}")
                return

        # 3. Delete tuned config if exists
        config_name = STAGE_MAP[stage]["config"]
        tuned_path = os.path.join("configs", f"tuned_{config_name}")
        if os.path.exists(tuned_path):
            try:
                os.remove(tuned_path)
                removed.append(tuned_path)
            except Exception:
                pass

        # 4. Delete tuning config
        config_path = f"logs/hp_tuning_{stage}_config.json"
        if os.path.exists(config_path):
            try:
                os.remove(config_path)
                removed.append(config_path)
            except Exception:
                pass

        handler.send_json({"ok": True, "stage": stage, "removed": removed, "count": len(removed)})
        return

    if path == "/api/tuning/apply":
        stage = str(body.get("stage", "")).upper()
        if stage not in STAGE_MAP:
            handler.send_error_json(400, f"Unknown stage: {stage}")
            return

        target = body.get("target", "new")  # "base" = overwrite base config, "new" = save as tuned_*

        db_path = f"logs/hp_tuning_{stage}.db"
        results = _read_study_results(db_path)
        if not results or not results.get("best_trial"):
            handler.send_error_json(404, f"No tuning results found for stage {stage}")
            return

        best_params = results["best_trial"]["params"]
        config_name = STAGE_MAP[stage]["config"]
        config_path = os.path.join("configs", config_name)

        try:
            base_cfg = json.loads(open(config_path, "r", encoding="utf-8").read())
        except Exception as e:
            handler.send_error_json(500, f"Cannot read base config: {e}")
            return

        # Apply best params
        applied = {}
        for k, v in best_params.items():
            if isinstance(v, tuple):
                v = list(v)
            old_val = base_cfg.get(k)
            base_cfg[k] = v
            applied[k] = {"old": old_val, "new": v}

        # Save
        if target == "base":
            save_path = config_path
        else:
            save_path = os.path.join("configs", f"tuned_{config_name}")

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(base_cfg, f, indent=2, ensure_ascii=False)
                f.write("\n")
            handler.send_json({
                "ok": True,
                "stage": stage,
                "target": target,
                "saved_to": save_path,
                "applied": applied,
                "best_val_loss": results["best_trial"]["value"],
            })
        except Exception as e:
            handler.send_error_json(500, f"Failed to save config: {e}")
        return

    handler.send_error_json(404, f"Unknown tuning endpoint: {path}")
