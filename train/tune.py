"""Generic Optuna HP tuning wrapper for all training stages.

Usage:
    python -m train.tune --stage A --n_trials 30 --max_steps 2000
    python -m train.tune --stage E --n_trials 20 --max_steps 1000 --params '["lr","wd","proj_lr_mult"]'
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

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
        # Optuna needs hashable choices — convert lists to tuples
        safe = []
        for c in choices:
            if isinstance(c, list):
                safe.append(tuple(c))
            else:
                safe.append(c)
        val = trial.suggest_categorical(name, safe)
        # Convert tuples back to lists for JSON config
        if isinstance(val, tuple):
            return list(val)
        return val
    return None


def _get_best_val_loss_from_jsonl(metrics_file: str, run_id_substr: str) -> float:
    """Read best val_loss from JSONL metrics after training completes."""
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
            if run_id_substr and run_id_substr not in str(rec.get("run_id", "")):
                continue
            val = rec.get("metric_value")
            if val is not None and val < best:
                best = val
    return best


def main():
    parser = argparse.ArgumentParser(description="Optuna HP tuning for micro-Omni")
    parser.add_argument("--stage", required=True, help="Stage letter: A-G")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps per trial")
    parser.add_argument("--params", type=str, default=None, help="JSON list of param names to tune (subset)")
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

    # Load search space from server API definitions
    from server.api.tuning import STAGE_SPECIFIC_SPACE, COMMON_TRAINING_SPACE
    search_space = STAGE_SPECIFIC_SPACE.get(stage, COMMON_TRAINING_SPACE)

    # Filter to requested params if specified
    if args.params:
        requested = set(json.loads(args.params))
        search_space = [s for s in search_space if s[0] in requested]
        print(f"Tuning subset: {[s[0] for s in search_space]}")

    # Load base config
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
    print(f"  DB: {db_path}")
    print(f"{'='*60}\n")

    # Import the training module's main function
    import importlib
    train_mod = importlib.import_module(module_name)
    train_main = train_mod.main

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)

        # Apply search space suggestions
        for name, kind, param_args in search_space:
            val = _suggest_param(trial, name, kind, param_args)
            if val is not None:
                cfg[name] = val

        # Override for tuning: fewer steps, unique save dir, skip checkpoints
        cfg["max_steps"] = args.max_steps
        cfg["checkpoint_freq"] = 999999
        cfg["save_dir"] = f"checkpoints/tune_{stage}/trial_{trial.number}"
        cfg["val_freq"] = min(cfg.get("val_freq", 100), 100)
        cfg["print_freq"] = 200
        os.makedirs(cfg["save_dir"], exist_ok=True)

        print(f"\n--- Trial {trial.number} ---")
        print(f"  Params: {trial.params}")

        try:
            # Run training
            result = train_main(cfg)

            # If main() returns val_loss directly, use it
            if isinstance(result, (int, float)) and result < 1e6:
                val_loss = float(result)
            else:
                # Fall back to reading JSONL
                val_loss = _get_best_val_loss_from_jsonl(
                    metrics_path,
                    run_id_substr=cfg["save_dir"],
                )

            if val_loss == float("inf"):
                # No val loss found — use a large penalty
                val_loss = 100.0

            print(f"  Result: val_loss={val_loss:.4f}")
            return val_loss

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return 100.0  # penalty for failed trials

    # Create or resume study
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
    print(f"  Best val loss: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # Save best config
    best_cfg = copy.deepcopy(base_cfg)
    for k, v in study.best_params.items():
        if isinstance(v, tuple):
            v = list(v)
        best_cfg[k] = v
    best_cfg_path = os.path.join("configs", f"tuned_{config_file}")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, ensure_ascii=False)
    print(f"Best config saved to: {best_cfg_path}")


if __name__ == "__main__":
    main()
