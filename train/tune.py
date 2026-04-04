"""Generic Optuna HP tuning wrapper for all training stages.

Supports multi-objective optimization on real test metrics (CER, perplexity, R@1, etc.)
using Optuna's Pareto-front optimization. Each selected metric becomes a separate objective.

Usage:
    python -m train.tune --stage A --n_trials 30 --max_steps 2000
    python -m train.tune --stage E --n_trials 20 --params '["lr","wd","proj_lr_mult"]'
    python -m train.tune --stage B --n_trials 20 --max_steps 3000 --metrics '["cer","wer"]'
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

# Fix Windows encoding for subprocess pipes (tqdm progress bars use Unicode)
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Stage → (training module function name, config file, metrics JSONL file)
STAGE_INFO = {
    "A": ("train.train_thinker", "synthetic_thinker.json", "train_thinker.jsonl"),
    "B": ("train.train_audio_enc", "synthetic_audio_enc.json", "train_audio_enc.jsonl"),
    "C": ("train.train_vision", "synthetic_vision.json", "train_vision.jsonl"),
    "D": ("train.train_talker", "synthetic_talker.json", "train_talker.jsonl"),
    "E": ("train.sft_omni", "synthetic_omni_sft.json", "sft_omni.jsonl"),
    "F": ("train.train_vocoder", "synthetic_vocoder.json", "train_vocoder.jsonl"),
    "G": ("train.train_ocr", "synthetic_ocr.json", "train_ocr.jsonl"),
}


def _suggest_param(trial, name: str, kind: str, args: tuple):
    """Map search space definition to Optuna trial suggestions."""
    if kind == "float_log":
        return trial.suggest_float(name, args[0], args[1], log=True)
    elif kind == "float":
        return trial.suggest_float(name, args[0], args[1])
    elif kind == "int":
        return trial.suggest_int(name, args[0], args[1])
    elif kind == "categorical":
        choices = args[0]
        safe = []
        for c in choices:
            if isinstance(c, list):
                safe.append(tuple(c))
            else:
                safe.append(c)
        val = trial.suggest_categorical(name, safe)
        if isinstance(val, tuple):
            return list(val)
        return val
    return None


def _get_best_val_loss_from_jsonl(metrics_file: str, run_id: str) -> float:
    """Read best val_loss from JSONL metrics for a specific run_id."""
    best = float("inf")
    if not os.path.exists(metrics_file):
        return best
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("metric_name") != "val_loss":
                continue
            if run_id and rec.get("run_id") != run_id:
                continue
            val = rec.get("metric_value")
            if val is not None and val < best:
                best = val
    return best


def _get_test_metrics_from_jsonl(metrics_file: str, metric_keys: list[str]) -> dict[str, float]:
    """Read test metrics from a test script's JSONL output."""
    results: dict[str, float] = {}
    full_path = os.path.join("logs", "metrics", metrics_file)
    if not os.path.exists(full_path):
        return results
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            name = rec.get("metric_name", "")
            if name in metric_keys and rec.get("phase") == "test":
                val = rec.get("metric_value")
                if val is not None:
                    results[name] = float(val)
    return results


def _run_test_for_trial(stage: str, checkpoint_dir: str, num_samples: int = 50) -> dict[str, float]:
    """Run the test script for a stage and return extracted metrics."""
    from server.api.tuning import STAGE_TEST_INFO, STAGE_METRICS
    test_info = STAGE_TEST_INFO.get(stage)
    if not test_info:
        return {}

    test_module = test_info["module"]
    test_jsonl = test_info["jsonl"]
    metric_keys = [m[0] for m in STAGE_METRICS.get(stage, []) if m[0] != "val_loss"]

    if not metric_keys:
        return {}

    import subprocess
    cmd = [
        sys.executable, "-m", test_module,
        "--checkpoint", checkpoint_dir,
        "--num_samples", str(num_samples),
    ]
    if stage == "C":
        cmd.append("--retrieval")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            print(f"  Test script failed (exit {result.returncode}): {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  Test script timed out after 300s")
        return {}
    except Exception as e:
        print(f"  Test script error: {e}")
        return {}

    return _get_test_metrics_from_jsonl(test_jsonl, metric_keys)


def _compute_objective(
    val_loss: float,
    test_metrics: dict[str, float],
    selected_metrics: list[str],
    stage: str,
) -> tuple[float, ...]:
    """Compute multi-objective values — one per selected metric.

    Returns a tuple of floats, all converted to "minimize" direction.
    Optuna's multi-objective mode finds the Pareto front across these.
    """
    from server.api.tuning import STAGE_METRICS

    metric_defs = {m[0]: m for m in STAGE_METRICS.get(stage, [])}
    values: list[float] = []

    for metric_key in selected_metrics:
        if metric_key == "val_loss":
            values.append(val_loss if val_loss < 100 else 10.0)
            continue

        if metric_key not in test_metrics:
            # Metric not available — use penalty
            values.append(100.0)
            continue

        value = test_metrics[metric_key]
        meta = metric_defs.get(metric_key)
        if not meta:
            values.append(value)
            continue

        direction = meta[2]  # "minimize" or "maximize"
        if direction == "maximize":
            value = 1.0 - value  # Convert to minimize

        values.append(value)

    if not values:
        return (val_loss if val_loss < 100 else 100.0,)

    return tuple(values)


def main():
    parser = argparse.ArgumentParser(description="Optuna HP tuning for micro-Omni")
    parser.add_argument("--stage", required=True, help="Stage letter: A-G")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--max_steps", type=int, default=500, help="Max training steps per trial")
    parser.add_argument("--params", type=str, default=None, help="JSON list of param names to tune (subset)")
    parser.add_argument("--metrics", type=str, default=None,
                       help="JSON list of metric keys to optimize (e.g. '[\"cer\",\"wer\"]'). "
                            "If not specified, uses val_loss only (legacy behavior).")
    parser.add_argument("--test_samples", type=int, default=50,
                       help="Number of samples for test evaluation per trial (default: 50)")
    parser.add_argument("--config", type=str, default=None, help="Override base config file")
    args = parser.parse_args()

    stage = args.stage.upper()
    if stage not in STAGE_INFO:
        print(f"Unknown stage: {stage}. Valid: {list(STAGE_INFO.keys())}")
        sys.exit(1)

    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    module_name, default_config, metrics_file = STAGE_INFO[stage]

    # Parse selected metrics
    selected_metrics: list[str] = []
    if args.metrics:
        selected_metrics = json.loads(args.metrics)
    use_test_metrics = bool(selected_metrics and any(m != "val_loss" for m in selected_metrics))
    is_multi_objective = len(selected_metrics) > 1

    if use_test_metrics:
        print(f"Optimizing metrics: {selected_metrics}")
        print(f"  Multi-objective: {is_multi_objective} ({len(selected_metrics)} objectives)")
        print(f"  Test samples per trial: {args.test_samples}")
    else:
        if not selected_metrics:
            selected_metrics = ["val_loss"]
        print(f"Optimizing: val_loss only (legacy mode)")

    # Load search space
    from server.api.tuning import STAGE_SPECIFIC_SPACE, COMMON_TRAINING_SPACE
    search_space = STAGE_SPECIFIC_SPACE.get(stage, COMMON_TRAINING_SPACE)

    if args.params:
        requested = set(json.loads(args.params))
        search_space = [s for s in search_space if s[0] in requested]
        print(f"Tuning subset: {[s[0] for s in search_space]}")

    config_file = args.config or default_config
    config_path = os.path.join("configs", config_file)
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)
    base_cfg = json.load(open(config_path, "r", encoding="utf-8"))

    metrics_path = os.path.join("logs", "metrics", metrics_file)
    db_path = os.path.join("logs", f"hp_tuning_{stage}.db")
    os.makedirs("logs", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  micro-Omni HP Tuning — Stage {stage}")
    print(f"  Config: {config_file}")
    print(f"  Trials: {args.n_trials}, Max steps/trial: {args.max_steps}")
    print(f"  Params: {[s[0] for s in search_space]}")
    print(f"  Metrics: {selected_metrics}")
    print(f"  DB: {db_path}")
    print(f"{'='*60}\n")

    import importlib
    train_mod = importlib.import_module(module_name)
    train_main = train_mod.main

    def objective(trial: optuna.Trial) -> float | tuple[float, ...]:
        cfg = copy.deepcopy(base_cfg)

        for name, kind, param_args in search_space:
            val = _suggest_param(trial, name, kind, param_args)
            if val is not None:
                cfg[name] = val

        cfg["max_steps"] = args.max_steps
        cfg["checkpoint_freq"] = 999999
        cfg["save_dir"] = f"checkpoints/tune_{stage}/trial_{trial.number}"
        cfg["val_freq"] = min(cfg.get("val_freq", 100), 100)
        cfg["print_freq"] = 200
        os.makedirs(cfg["save_dir"], exist_ok=True)

        print(f"\n--- Trial {trial.number} ---")
        print(f"  Params: {trial.params}")

        try:
            from omni.training_utils import build_run_id
            script_name = module_name.split(".")[-1] + ".py"
            trial_run_id = build_run_id(script_name, None, cfg["save_dir"])

            result = train_main(cfg)

            if isinstance(result, (int, float)) and result < 1e6:
                val_loss = float(result)
            else:
                # Try with run_id first, then without (JSONL may have timing issues)
                val_loss = _get_best_val_loss_from_jsonl(metrics_path, run_id=trial_run_id)
                if val_loss == float("inf"):
                    val_loss = _get_best_val_loss_from_jsonl(metrics_path, run_id="")
            if val_loss == float("inf"):
                val_loss = 100.0

            # Run test script if optimizing real metrics
            test_metrics: dict[str, float] = {}
            if use_test_metrics:
                print(f"  Running test evaluation ({args.test_samples} samples)...")
                test_metrics = _run_test_for_trial(
                    stage, cfg["save_dir"], num_samples=args.test_samples
                )
                if test_metrics:
                    print(f"  Test metrics: {test_metrics}")
                else:
                    print(f"  Warning: no test metrics returned, falling back to val_loss")

            # Store per-metric values as user attributes for DB readback
            trial.set_user_attr("val_loss", val_loss)
            for k, v in test_metrics.items():
                trial.set_user_attr(k, v)

            # Compute objectives
            obj_values = _compute_objective(val_loss, test_metrics, selected_metrics, stage)

            if is_multi_objective:
                print(f"  Result: objectives={obj_values} (val_loss={val_loss:.4f}, test={test_metrics})")
                return obj_values
            else:
                print(f"  Result: objective={obj_values[0]:.4f} (val_loss={val_loss:.4f}, test={test_metrics})")
                return obj_values[0]

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            if is_multi_objective:
                return tuple(100.0 for _ in selected_metrics)
            return 100.0

    # Create or resume study
    if is_multi_objective:
        # Multi-objective: one direction per metric (all minimize after conversion)
        directions = ["minimize"] * len(selected_metrics)
        study = optuna.create_study(
            study_name=f"tune_{stage}",
            directions=directions,
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
    else:
        study = optuna.create_study(
            study_name=f"tune_{stage}",
            direction="minimize",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=300,
                interval_steps=100,
            ),
        )

    study.optimize(objective, n_trials=args.n_trials)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Tuning Complete — Stage {stage}")
    print(f"  Total trials: {len(study.trials)}")
    if is_multi_objective:
        pareto = study.best_trials
        print(f"  Pareto-optimal trials: {len(pareto)}")
        for i, t in enumerate(pareto[:5]):
            print(f"    #{t.number}: values={t.values}, user_attrs={t.user_attrs}")
    else:
        print(f"  Best objective: {study.best_value:.4f}")
    print(f"  Best params:")
    best_params = study.best_trials[0].params if is_multi_objective else study.best_params
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  Metrics optimized: {selected_metrics}")
    print(f"{'='*60}\n")

    # Save best config (use first Pareto-optimal trial for multi-objective)
    best_cfg = copy.deepcopy(base_cfg)
    for k, v in best_params.items():
        if isinstance(v, tuple):
            v = list(v)
        best_cfg[k] = v
    best_cfg_path = os.path.join("configs", f"tuned_{config_file}")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, ensure_ascii=False)
    print(f"Best config saved to: {best_cfg_path}")


if __name__ == "__main__":
    main()
