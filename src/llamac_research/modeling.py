"""Model training and evaluation harness for LLaMAC tabular features."""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import polars as pl

from . import __version__
from .device import BackendSelection, DeviceRequest, select_lightgbm_device, select_xgboost_device
from .features import read_feature_table
from .labels import (
    EMOTION_IDS,
    SELF_REPORT_COLUMNS,
    filter_official_valid_trials,
    resolve_target,
)
from .metrics import align_proba_columns, compute_classification_metrics, metrics_summary_line

FeatureSet = Literal["all", "ppg"]
SplitStrategy = Literal["grouped", "stratified"]
ModelName = Literal[
    "lightgbm",
    "logistic_regression",
    "svc_rbf",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
]

MODEL_NAMES: tuple[str, ...] = (
    "lightgbm",
    "logistic_regression",
    "svc_rbf",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable model-evaluation configuration."""

    feature_path: str
    model_name: str = "lightgbm"
    feature_set: str = "all"
    target: str = "reported"
    split_strategy: str = "grouped"
    n_splits: int = 5
    seed: int = 42
    device: str = "auto"
    apply_official_exclusions: bool = False
    max_rows: int | None = None
    output_dir: str = "artifacts/results"
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureMatrix:
    """Prepared raw feature matrix before fold-specific imputation."""

    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    label_distribution: dict[str, int]


def git_commit() -> str | None:
    """Return current git commit hash when available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_feature_columns(frame: pl.DataFrame, feature_set: FeatureSet = "all") -> list[str]:
    """Return candidate numeric feature columns with no questionnaire leakage."""
    leakage = {"SubjectID", "Trial", *SELF_REPORT_COLUMNS}
    if feature_set == "ppg":
        return [c for c in frame.columns if c.startswith("Band_PPG_")]
    if feature_set != "all":
        raise ValueError(f"Unsupported feature_set={feature_set!r}; use 'all' or 'ppg'.")
    out: list[str] = []
    for col, dtype in zip(frame.columns, frame.dtypes, strict=True):
        if col in leakage:
            continue
        if dtype.is_numeric():
            out.append(col)
    return out


def load_feature_matrix(
    feature_path: str | Path,
    *,
    feature_set: FeatureSet,
    target: str,
    apply_official_exclusions: bool = False,
    max_rows: int | None = None,
) -> FeatureMatrix:
    """Load feature table and resolve X/y/groups arrays."""
    target_spec = resolve_target(target)
    frame = read_feature_table(feature_path)
    if apply_official_exclusions:
        frame = filter_official_valid_trials(frame)
    if target_spec.column not in frame.columns:
        raise ValueError(f"{target_spec.column} is missing from {feature_path}; rebuild features with target columns.")
    if "SubjectID" not in frame.columns:
        raise ValueError("SubjectID column is required for participant-grouped splits")
    feature_names = select_feature_columns(frame, feature_set=feature_set)
    if not feature_names:
        raise ValueError(f"No feature columns found for feature_set={feature_set!r}")

    frame = frame.with_columns(pl.col(target_spec.column).cast(pl.Int64, strict=False))
    frame = frame.filter(pl.col(target_spec.column).is_in(EMOTION_IDS))
    if max_rows is not None:
        frame = frame.head(max_rows)
    feature_frame = frame.select([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in feature_names])
    x = feature_frame.to_numpy().astype(float, copy=False)
    y = frame[target_spec.column].to_numpy().astype(int, copy=False)
    groups = frame["SubjectID"].cast(pl.Utf8).to_numpy()
    labels, counts = np.unique(y, return_counts=True)
    distribution = {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)}
    return FeatureMatrix(x=x, y=y, groups=groups, feature_names=feature_names, label_distribution=distribution)


def _fold_transform(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_nan_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Drop unstable columns and median-impute using training-fold statistics only."""
    x_tr = np.asarray(x_train, dtype=float).copy()
    x_te = np.asarray(x_test, dtype=float).copy()
    x_tr[~np.isfinite(x_tr)] = np.nan
    x_te[~np.isfinite(x_te)] = np.nan
    nan_ratio = np.mean(np.isnan(x_tr), axis=0)
    keep = nan_ratio < max_nan_ratio
    # Drop constants after considering finite training values.
    for idx in range(x_tr.shape[1]):
        if not keep[idx]:
            continue
        finite = x_tr[:, idx][np.isfinite(x_tr[:, idx])]
        if finite.size == 0 or np.unique(finite).size <= 1:
            keep[idx] = False
    if not np.any(keep):
        raise ValueError("All feature columns were dropped by fold preprocessing")
    x_tr = x_tr[:, keep]
    x_te = x_te[:, keep]
    selected_names = [name for name, flag in zip(feature_names, keep, strict=True) if flag]
    medians = np.nanmedian(x_tr, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train_nan = np.isnan(x_tr)
    test_nan = np.isnan(x_te)
    x_tr[train_nan] = np.take(medians, np.where(train_nan)[1])
    x_te[test_nan] = np.take(medians, np.where(test_nan)[1])
    info = {
        "input_features": len(feature_names),
        "selected_features": len(selected_names),
        "dropped_features": int(len(feature_names) - len(selected_names)),
    }
    return x_tr, x_te, selected_names, info


def make_splitter(strategy: SplitStrategy, n_splits: int, seed: int):
    """Create sklearn splitter."""
    if strategy == "grouped":
        from sklearn.model_selection import StratifiedGroupKFold

        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "stratified":
        from sklearn.model_selection import StratifiedKFold

        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unsupported split_strategy={strategy!r}")


def split_indices(matrix: FeatureMatrix, strategy: SplitStrategy, n_splits: int, seed: int):
    splitter = make_splitter(strategy, n_splits=n_splits, seed=seed)
    if strategy == "grouped":
        yield from splitter.split(matrix.x, matrix.y, matrix.groups)
    else:
        yield from splitter.split(matrix.x, matrix.y)


def _backend_for_model(model_name: str, requested_device: DeviceRequest) -> BackendSelection:
    if model_name == "lightgbm":
        return select_lightgbm_device(requested_device)
    if model_name == "xgboost":
        return select_xgboost_device(requested_device)
    return BackendSelection(requested=requested_device, selected="cpu", backend=model_name)


def create_estimator(
    model_name: ModelName,
    *,
    seed: int,
    device_selection: BackendSelection,
    params: dict[str, Any] | None = None,
):
    """Instantiate a supported classifier."""
    params = dict(params or {})
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        base = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.1,
            "min_child_samples": 20,
            "min_split_gain": 0.0,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 1,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
            "device_type": device_selection.selected,
        }
        base.update(params)
        return LGBMClassifier(**base)

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = {"max_iter": 2000, "class_weight": "balanced", "random_state": seed}
        base.update(params)
        return make_pipeline(StandardScaler(), LogisticRegression(**base))

    if model_name == "svc_rbf":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        base = {"C": 3.0, "gamma": "scale", "class_weight": "balanced", "probability": True, "random_state": seed}
        base.update(params)
        return make_pipeline(StandardScaler(), SVC(**base))

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        base = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": seed,
            "n_jobs": -1,
        }
        base.update(params)
        return RandomForestClassifier(**base)

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        base = {
            "n_estimators": 700,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
        }
        base.update(params)
        return ExtraTreesClassifier(**base)

    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        base = {"learning_rate": 0.05, "max_iter": 300, "l2_regularization": 0.01, "random_state": seed}
        base.update(params)
        return HistGradientBoostingClassifier(**base)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        base = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": seed,
            "n_jobs": 1,
            "device": device_selection.selected,
        }
        base.update(params)
        return XGBClassifier(**base)

    raise ValueError(f"Unsupported model_name={model_name!r}")


def _predict_scores(model: Any, x: np.ndarray, labels: Sequence[int] = EMOTION_IDS) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        pred = model.predict(x)
    pred_arr = np.asarray(pred, dtype=int)
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            proba = np.asarray(model.predict_proba(x), dtype=float)
        classes = getattr(model, "classes_", labels)
        if not hasattr(model, "classes_") and hasattr(model, "named_steps"):
            # sklearn Pipeline exposes classes_ on the final estimator only in some versions.
            classes = getattr(model.steps[-1][1], "classes_", labels)
        scores = align_proba_columns(classes, proba, labels=labels)
    else:
        scores = None
    if scores is None:
        scores = align_proba_columns(labels, np.eye(len(labels))[np.searchsorted(labels, pred_arr)], labels=labels)
    return pred_arr, scores


def run_cross_validated_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a CV experiment and return a JSON-serializable result bundle."""
    start = time.time()
    if config.model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model {config.model_name!r}; choices={MODEL_NAMES}")
    matrix = load_feature_matrix(
        config.feature_path,
        feature_set=config.feature_set,  # type: ignore[arg-type]
        target=config.target,
        apply_official_exclusions=config.apply_official_exclusions,
        max_rows=config.max_rows,
    )
    if len(np.unique(matrix.y)) < 2:
        raise ValueError("Need at least two target classes for classification")
    backend = _backend_for_model(config.model_name, config.device)  # type: ignore[arg-type]
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    score_all: list[np.ndarray] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        split_indices(matrix, config.split_strategy, config.n_splits, config.seed), start=1
    ):
        x_train, x_test, selected_names, transform_info = _fold_transform(
            matrix.x[train_idx], matrix.x[test_idx], matrix.feature_names
        )
        y_train = matrix.y[train_idx]
        y_test = matrix.y[test_idx]
        model = create_estimator(
            config.model_name,  # type: ignore[arg-type]
            seed=config.seed + fold_idx,
            device_selection=backend,
            params=config.model_params,
        )
        fit_start = time.time()
        if config.model_name == "lightgbm":
            try:
                from lightgbm.callback import early_stopping, log_evaluation

                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_test, y_test)],
                    eval_metric="multi_logloss",
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
                )
            except TypeError:
                model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train)
        y_pred, scores = _predict_scores(model, x_test)
        fold_metrics = compute_classification_metrics(y_test, y_pred, scores).to_dict()
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        score_all.append(scores)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_rows": int(train_idx.size),
                "test_rows": int(test_idx.size),
                "train_groups": int(np.unique(matrix.groups[train_idx]).size),
                "test_groups": int(np.unique(matrix.groups[test_idx]).size),
                "fit_seconds": round(time.time() - fit_start, 3),
                **transform_info,
                "metrics": fold_metrics,
            }
        )
        print(f"[fold {fold_idx}] {metrics_summary_line(fold_metrics)}", flush=True)

    all_scores = np.vstack(score_all)
    overall_metrics = compute_classification_metrics(y_true_all, y_pred_all, all_scores).to_dict()
    elapsed = time.time() - start
    result = {
        "schema_version": 1,
        "created_at_unix": int(time.time()),
        "elapsed_seconds": round(elapsed, 3),
        "config": asdict(config),
        "environment": {
            "llamac_research_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit(),
            "feature_file_sha256": file_sha256(config.feature_path),
        },
        "backend": backend.to_dict(),
        "data": {
            "rows": int(matrix.y.size),
            "candidate_features": int(len(matrix.feature_names)),
            "label_distribution": matrix.label_distribution,
            "groups": int(np.unique(matrix.groups).size),
        },
        "folds": fold_summaries,
        "metrics": overall_metrics,
    }
    return result


def default_result_path(config: ExperimentConfig, result: dict[str, Any]) -> Path:
    """Build deterministic-ish output path from config and timestamp."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(result["created_at_unix"]))
    official = "official" if config.apply_official_exclusions else "allsubjects"
    name = f"{stamp}_{config.model_name}_{config.feature_set}_{config.target}_{config.split_strategy}_{official}.json"
    return Path(config.output_dir) / name


def save_experiment_result(result: dict[str, Any], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def run_and_save(config: ExperimentConfig, output_path: str | Path | None = None) -> Path:
    result = run_cross_validated_experiment(config)
    out = Path(output_path) if output_path is not None else default_result_path(config, result)
    save_experiment_result(result, out)
    print(f"[result] {metrics_summary_line(result['metrics'])}", flush=True)
    print(f"[saved] {out}", flush=True)
    return out
