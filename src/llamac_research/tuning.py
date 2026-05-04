"""Optuna tuning helpers for LLaMAC tabular model families."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .metrics import metrics_summary_line
from .modeling import ExperimentConfig, MODEL_NAMES, run_cross_validated_experiment, save_experiment_result


@dataclass(frozen=True)
class TuningConfig:
    """Serializable Optuna tuning configuration."""

    feature_path: str
    models: list[str]
    feature_set: str = "ppg"
    target: str = "reported"
    split_strategy: str = "grouped"
    n_splits: int = 5
    seed: int = 42
    device: str = "auto"
    apply_official_exclusions: bool = False
    n_trials: int = 20
    timeout_seconds: int | None = None
    metric: str = "macro_f1"
    output_dir: str = "artifacts/optuna"
    storage: str | None = None
    study_prefix: str = "llamac"
    max_rows: int | None = None


def sample_model_params(trial: Any, model_name: str) -> dict[str, Any]:
    """Return model-specific Optuna search-space parameters."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": "lbfgs",
        }
    if model_name == "svc_rbf":
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 700),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        }
    raise ValueError(f"Unsupported model_name={model_name!r}")


def _study_storage(config: TuningConfig, model_name: str) -> str | None:
    if config.storage:
        return config.storage
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{out / f'{config.study_prefix}_{model_name}_{config.feature_set}_{config.target}.db'}"


def run_tuning(config: TuningConfig) -> dict[str, Any]:
    """Run Optuna studies for one or more model families."""
    import optuna

    invalid = [model for model in config.models if model not in MODEL_NAMES]
    if invalid:
        raise ValueError(f"Unsupported models {invalid}; choices={MODEL_NAMES}")

    started = time.time()
    summaries: list[dict[str, Any]] = []
    final_results: list[dict[str, Any]] = []
    for model_name in config.models:
        storage = _study_storage(config, model_name)
        study_name = f"{config.study_prefix}_{model_name}_{config.feature_set}_{config.target}_{config.split_strategy}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=config.seed),
        )

        def objective(trial: Any) -> float:
            params = sample_model_params(trial, model_name)
            exp = ExperimentConfig(
                feature_path=config.feature_path,
                model_name=model_name,
                feature_set=config.feature_set,
                target=config.target,
                split_strategy=config.split_strategy,
                n_splits=config.n_splits,
                seed=config.seed,
                device=config.device,
                apply_official_exclusions=config.apply_official_exclusions,
                max_rows=config.max_rows,
                output_dir=str(Path(config.output_dir) / "trial-results"),
                model_params=params,
            )
            result = run_cross_validated_experiment(exp)
            value = float(result["metrics"][config.metric])
            trial.set_user_attr("metrics", result["metrics"])
            trial.set_user_attr("backend", result["backend"])
            trial.set_user_attr("data", result["data"])
            return value

        study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds, gc_after_trial=True)
        best_params = dict(study.best_trial.params)
        locked = ExperimentConfig(
            feature_path=config.feature_path,
            model_name=model_name,
            feature_set=config.feature_set,
            target=config.target,
            split_strategy=config.split_strategy,
            n_splits=config.n_splits,
            seed=config.seed,
            device=config.device,
            apply_official_exclusions=config.apply_official_exclusions,
            max_rows=config.max_rows,
            output_dir=str(Path(config.output_dir) / "locked-results"),
            model_params=best_params,
        )
        locked_result = run_cross_validated_experiment(locked)
        locked_path = Path(config.output_dir) / "locked-results" / (
            f"{model_name}_{config.feature_set}_{config.target}_{config.split_strategy}_best.json"
        )
        save_experiment_result(locked_result, locked_path)
        summary = {
            "model_name": model_name,
            "study_name": study.study_name,
            "storage": storage,
            "n_trials_total": len(study.trials),
            "best_trial_number": study.best_trial.number,
            "best_value": float(study.best_value),
            "metric": config.metric,
            "best_params": best_params,
            "locked_result_path": str(locked_path),
            "locked_metrics": locked_result["metrics"],
        }
        print(
            f"[optuna:{model_name}] best_{config.metric}={study.best_value:.4f} "
            f"locked {metrics_summary_line(locked_result['metrics'])}",
            flush=True,
        )
        summaries.append(summary)
        final_results.append(locked_result)

    out = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started, 3),
        "config": asdict(config),
        "studies": summaries,
    }
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{config.study_prefix}_{config.feature_set}_{config.target}_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    out["summary_path"] = str(summary_path)
    print(f"[optuna:summary] {summary_path}", flush=True)
    return out
