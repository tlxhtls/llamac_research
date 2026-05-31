#!/usr/bin/env python
"""Validation-only strict PPG emotion candidate and ensemble search."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import platform
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llamac_research import __version__  # noqa: E402
from llamac_research.features import _ppg_features  # noqa: E402
from llamac_research.labels import EMOTION_ID_TO_LABEL, EMOTION_IDS  # noqa: E402
from llamac_research.metrics import metrics_summary_line  # noqa: E402
from llamac_research.waveform_dnn import (  # noqa: E402
    RATING_TARGETS,
    WaveformExample,
    create_rating_model,
    git_commit,
    load_ppg_rating_examples,
    make_subject_split,
    select_torch_device,
)
from scripts.train_ppg_waveform_emotion import (  # noqa: E402
    PpgEmotionWindowDataset,
    _emotion_arrays,
    _metrics_from_probs,
    _predict_probs,
    _prior_baseline,
)


METRIC_KEYS = (
    "top1_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "cohen_kappa",
)


@dataclass(frozen=True)
class SearchConfig:
    data_root: str = "data/extracted"
    output_dir: str = "artifacts/results"
    label_column: Literal["ReportedType", "IntendedType"] = "ReportedType"
    result_glob: str = "ppg_*_reported_emotion_result_*.json"
    window_seconds: float = 30.0
    train_windows_per_trial: int = 4
    eval_windows_per_trial: int = 1
    split_seed: int = 42
    seed: int = 777
    val_subject_fraction: float = 0.15
    test_subject_fraction: float = 0.15
    max_subjects: int | None = None
    batch_size: int = 512
    num_workers: int = 4
    device: Literal["auto", "cuda", "cpu"] = "auto"
    tabular_device: Literal["cpu", "cuda"] = "cpu"
    include_dnn: bool = True
    include_tabular: bool = True
    include_ensembles: bool = True
    allow_amp_results: bool = False
    tabular_models: tuple[str, ...] = (
        "logistic_regression",
        "svc_rbf",
        "extra_trees",
        "random_forest",
        "hist_gradient_boosting",
        "lightgbm",
        "xgboost",
        "catboost",
    )
    rich_features: bool = True
    max_ensemble_members: int = 4
    max_ensemble_pool: int = 10
    max_evaluated_candidates: int = 60
    max_eligible_candidates: int = 40
    no_improvement_patience: int = 8
    improvement_delta: float = 0.005
    max_improvements: int = 5
    min_macro_f1: float = 0.15
    max_prediction_share: float = 0.75
    wall_clock_limit_seconds: float = 4 * 60 * 60
    tabular_model_timeout_seconds: float = 5 * 60
    target_top1: float = 0.30
    target_top3: float = 0.70
    target_macro_f1: float = 0.23
    target_kappa: float = 0.04


class Candidate:
    def __init__(
        self,
        *,
        name: str,
        family: str,
        source: str,
        val_probs: np.ndarray,
        val_metrics: dict[str, Any],
        eligible: bool,
        eligibility_reason: str,
        metadata: dict[str, Any],
        test_fn: Callable[[], np.ndarray] | None,
    ) -> None:
        self.name = name
        self.family = family
        self.source = source
        self.val_probs = val_probs
        self.val_metrics = val_metrics
        self.eligible = eligible
        self.eligibility_reason = eligibility_reason
        self.metadata = metadata
        self._test_fn = test_fn
        self._test_probs: np.ndarray | None = None

    def test_probs(self) -> np.ndarray:
        if self._test_probs is None:
            if self._test_fn is None:
                raise RuntimeError(f"No test predictor registered for {self.name}")
            self._test_probs = self._test_fn()
        return self._test_probs

    def row(self, *, include_probs: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "source": self.source,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "val_metrics": self.val_metrics,
            "metadata": self.metadata,
        }
        if include_probs:
            out["val_probs"] = self.val_probs.tolist()
        return out


class StopTracker:
    def __init__(self, config: SearchConfig, *, initial_candidates: Iterable[Candidate] = ()) -> None:
        self.config = config
        self.started_at = time.time()
        self.evaluated_candidates = 0
        self.eligible_candidates = 0
        self.no_improvement_count = 0
        self.improvement_count = 0
        self.stop_reason: str | None = None
        self.events: list[dict[str, Any]] = []
        self.best_val_top1 = -math.inf
        self.best_val_top3 = -math.inf
        for candidate in initial_candidates:
            self.seed(candidate)

    def seed(self, candidate: Candidate) -> None:
        self.evaluated_candidates += 1
        if not candidate.eligible:
            return
        self.eligible_candidates += 1
        self.best_val_top1 = max(self.best_val_top1, float(candidate.val_metrics["top1_accuracy"]))
        self.best_val_top3 = max(self.best_val_top3, float(candidate.val_metrics["top3_accuracy"]))

    def observe(self, candidate: Candidate) -> bool:
        self.evaluated_candidates += 1
        if not candidate.eligible:
            self._check_hard_limits()
            return self.stop_reason is not None

        self.eligible_candidates += 1
        val_top1 = float(candidate.val_metrics["top1_accuracy"])
        val_top3 = float(candidate.val_metrics["top3_accuracy"])
        improved_top1 = val_top1 >= self.best_val_top1 + self.config.improvement_delta
        improved_top3 = val_top3 >= self.best_val_top3 + self.config.improvement_delta
        if improved_top1 or improved_top3:
            self.improvement_count += 1
            self.no_improvement_count = 0
            self.events.append(
                {
                    "candidate": candidate.name,
                    "improved_top1": improved_top1,
                    "improved_top3": improved_top3,
                    "previous_best_top1": self.best_val_top1,
                    "previous_best_top3": self.best_val_top3,
                    "new_top1": val_top1,
                    "new_top3": val_top3,
                }
            )
            self.best_val_top1 = max(self.best_val_top1, val_top1)
            self.best_val_top3 = max(self.best_val_top3, val_top3)
        else:
            self.no_improvement_count += 1

        if self.improvement_count >= self.config.max_improvements:
            self.stop_reason = f"max_improvements={self.config.max_improvements}"
        elif self.no_improvement_count >= self.config.no_improvement_patience:
            self.stop_reason = f"no_improvement_patience={self.config.no_improvement_patience}"
        elif (
            (val_top1 >= self.config.target_top1 or val_top3 >= self.config.target_top3)
            and float(candidate.val_metrics["macro_f1"]) >= self.config.target_macro_f1
            and float(candidate.val_metrics["cohen_kappa"]) > self.config.target_kappa
        ):
            self.stop_reason = "target_reached_freeze"
        self._check_hard_limits()
        return self.stop_reason is not None

    def _check_hard_limits(self) -> None:
        if self.stop_reason is not None:
            return
        if self.evaluated_candidates >= self.config.max_evaluated_candidates:
            self.stop_reason = f"max_evaluated_candidates={self.config.max_evaluated_candidates}"
        elif self.eligible_candidates >= self.config.max_eligible_candidates:
            self.stop_reason = f"max_eligible_candidates={self.config.max_eligible_candidates}"
        elif time.time() - self.started_at >= self.config.wall_clock_limit_seconds:
            self.stop_reason = f"wall_clock_limit_seconds={self.config.wall_clock_limit_seconds:g}"

    def state(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "evaluated_candidates": self.evaluated_candidates,
            "eligible_candidates": self.eligible_candidates,
            "no_improvement_count": self.no_improvement_count,
            "improvement_count": self.improvement_count,
            "best_val_top1": self.best_val_top1,
            "best_val_top3": self.best_val_top3,
            "events": self.events,
            "elapsed_seconds": time.time() - self.started_at,
        }


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(math.isfinite(float(metrics[key])) for key in METRIC_KEYS)


def _prediction_share(metrics: dict[str, Any]) -> float:
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    total = float(cm.sum())
    if total <= 0:
        return 1.0
    return float(cm.sum(axis=0).max() / total)


def _eligibility(metrics: dict[str, Any], config: SearchConfig) -> tuple[bool, str]:
    if not _finite_metrics(metrics):
        return False, "nonfinite_metric"
    if float(metrics["macro_f1"]) < config.min_macro_f1:
        return False, f"macro_f1<{config.min_macro_f1:g}"
    max_share = _prediction_share(metrics)
    if max_share > config.max_prediction_share:
        return False, f"prediction_share>{config.max_prediction_share:g}"
    return True, "eligible"


def _labels_to_names(labels: Iterable[int]) -> list[str]:
    return [EMOTION_ID_TO_LABEL[int(label)] for label in labels]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_result_ok(result: dict[str, Any], path: Path, config: SearchConfig) -> tuple[bool, str]:
    if result.get("task") != "emotion_classification":
        return False, "not_emotion_classification"
    if result.get("target") != config.label_column:
        return False, "different_target"
    cfg = result.get("config", {})
    strict = result.get("strict_conditions", {})
    if config.max_subjects is None and cfg.get("max_subjects") is not None:
        return False, "max_subjects_smoke"
    if not config.allow_amp_results and bool(cfg.get("amp", False)):
        return False, "amp_result_excluded"
    if not strict.get("ppg_only", False):
        return False, "not_ppg_only"
    if strict.get("full_signal_or_trial_aggregation", True):
        return False, "full_signal_or_trial_aggregation"
    if int(cfg.get("eval_windows_per_trial", -1)) != 1:
        return False, "eval_windows_per_trial_not_1"
    if float(cfg.get("eval_window_seconds", -1.0)) != float(config.window_seconds):
        return False, "eval_window_seconds_mismatch"
    checkpoint_path = Path(result.get("checkpoint_path", ""))
    if not checkpoint_path.exists():
        return False, f"missing_checkpoint:{checkpoint_path}"
    if not Path(path).exists():
        return False, "missing_result"
    return True, "ok"


def _make_dnn_predictor(
    *,
    result: dict[str, Any],
    examples: list[WaveformExample],
    labels: np.ndarray,
    device: str,
    batch_size: int,
    num_workers: int,
) -> Callable[[np.ndarray], np.ndarray]:
    cfg = result["config"]
    checkpoint_path = Path(result["checkpoint_path"])
    model_arch = str(result["model_arch"])

    def predict(indices: np.ndarray) -> np.ndarray:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = create_rating_model(
            model_arch,
            n_outputs=len(EMOTION_IDS),
            shared_dropout=float(cfg.get("shared_dropout", 0.25)),
            head_dropout=float(cfg.get("head_dropout", 0.20)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        ds = PpgEmotionWindowDataset(
            examples,
            indices,
            labels,
            label_column=str(cfg.get("label_column", result["target"])),
            target_length=int(cfg.get("target_length", 1920)),
            min_window_seconds=float(cfg.get("min_window_seconds", config_window_seconds(cfg))),
            max_window_seconds=float(cfg.get("max_window_seconds", config_window_seconds(cfg))),
            eval_window_seconds=float(cfg.get("eval_window_seconds", config_window_seconds(cfg))),
            train_window_anchor=str(cfg.get("train_window_anchor", "random")),  # type: ignore[arg-type]
            input_normalization=str(cfg.get("input_normalization", "robust")),  # type: ignore[arg-type]
            input_clip_value=cfg.get("input_clip_value", None),
            windows_per_trial=int(cfg.get("eval_windows_per_trial", 1)),
            train=False,
            seed=int(cfg.get("seed", 777)),
        )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == "cuda",
            persistent_workers=num_workers > 0,
        )
        probs = _predict_probs(model, loader, device=device)
        del checkpoint, model, ds, loader
        if device == "cuda":
            torch.cuda.empty_cache()
        return probs

    return predict


def config_window_seconds(cfg: dict[str, Any]) -> float:
    return float(cfg.get("eval_window_seconds", cfg.get("max_window_seconds", 30.0)))


def load_dnn_candidates(
    *,
    config: SearchConfig,
    examples: list[WaveformExample],
    labels: np.ndarray,
    split_val: np.ndarray,
    split_test: np.ndarray,
    device: str,
) -> list[Candidate]:
    if not config.include_dnn:
        return []
    output_dir = Path(config.output_dir)
    paths = sorted(output_dir.glob(config.result_glob))
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for path in paths:
        result = _load_json(path)
        ok, reason = _strict_result_ok(result, path, config)
        if not ok:
            continue
        name = f"dnn:{result['model_arch']}:{path.stem.rsplit('_', 1)[-1]}"
        if name in seen:
            continue
        seen.add(name)
        predictor = _make_dnn_predictor(
            result=result,
            examples=examples,
            labels=labels,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        val_probs = predictor(split_val)
        val_metrics = _metrics_from_probs(labels[split_val], val_probs)
        eligible, eligibility_reason = _eligibility(val_metrics, config)
        candidates.append(
            Candidate(
                name=name,
                family="dnn",
                source=str(path),
                val_probs=val_probs,
                val_metrics=val_metrics,
                eligible=eligible,
                eligibility_reason=eligibility_reason,
                metadata={
                    "model_arch": result["model_arch"],
                    "checkpoint_path": result["checkpoint_path"],
                    "result_path": str(path),
                    "baseline_result_test_metrics_available": True,
                    "selection_note": "validation metrics recomputed from checkpoint; test deferred until selected",
                },
                test_fn=lambda predictor=predictor: predictor(split_test),
            )
        )
    return candidates


def _crop_ppg_with_time(
    example: WaveformExample,
    *,
    window_seconds: float,
    start_fraction: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    y = example.ppg
    if y.size == 0:
        return np.zeros(1, dtype=np.float32), None
    fraction = min(max(float(start_fraction), 0.0), 1.0)
    if example.ppg_time is not None and example.ppg_time.size == y.size:
        t = example.ppg_time.astype(np.float64, copy=False)
        finite = np.isfinite(t)
        if finite.sum() >= 2:
            start_t = float(np.nanmin(t))
            end_t = float(np.nanmax(t))
            available = max(0.0, end_t - start_t)
            duration = min(float(window_seconds), available) if available > 0 else float(window_seconds)
            lo = start_t + max(0.0, available - duration) * fraction
            hi = lo + duration
            mask = (t >= lo) & (t <= hi)
            if mask.sum() >= 16:
                return y[mask].astype(np.float32, copy=False), t[mask].astype(np.float64, copy=False)

    n = y.size
    samples = max(16, min(n, int(round(n * min(float(window_seconds), 60.0) / 60.0))))
    start = int(round((n - samples) * fraction)) if n > samples else 0
    values = y[start : start + samples].astype(np.float32, copy=False)
    if example.ppg_time is not None and example.ppg_time.size == y.size:
        return values, example.ppg_time[start : start + samples].astype(np.float64, copy=False)
    return values, None


def _ppg_window_feature_row(
    example: WaveformExample,
    *,
    window_seconds: float,
    start_fraction: float,
    rich: bool,
) -> dict[str, float]:
    values, time_values = _crop_ppg_with_time(example, window_seconds=window_seconds, start_fraction=start_fraction)
    data: dict[str, Any] = {"PPG": values}
    if time_values is not None:
        data["PPG_Time"] = time_values
    df = pl.DataFrame(data)
    info: dict[str, Any] = {
        "window_seconds_requested": float(window_seconds),
        "window_start_fraction": float(start_fraction),
        "window_sample_count": int(values.size),
    }
    _ppg_features(df, info, rich=rich)
    out: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out


def _fractions(count: int) -> list[float]:
    if count <= 1:
        return [1.0]
    return [float(x) for x in np.linspace(0.0, 1.0, num=count)]


def _build_feature_rows(
    examples: list[WaveformExample],
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    window_seconds: float,
    windows_per_trial: int,
    rich: bool,
) -> tuple[list[dict[str, float]], np.ndarray]:
    rows: list[dict[str, float]] = []
    y: list[int] = []
    for idx in indices:
        example = examples[int(idx)]
        for fraction in _fractions(windows_per_trial):
            rows.append(
                _ppg_window_feature_row(
                    example,
                    window_seconds=window_seconds,
                    start_fraction=fraction,
                    rich=rich,
                )
            )
            y.append(int(labels[int(idx)]))
    return rows, np.asarray(y, dtype=np.int64)


def _matrix_from_rows(
    rows: list[dict[str, float]],
    *,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if feature_names is None:
        feature_names = sorted({key for row in rows for key in row.keys()})
    x = np.empty((len(rows), len(feature_names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(feature_names):
            value = row.get(name, math.nan)
            x[row_idx, col_idx] = float(value) if math.isfinite(float(value)) else np.nan
    return x, feature_names


def _sample_weight_balanced(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=len(EMOTION_IDS)).astype(np.float64)
    counts[counts <= 0] = 1.0
    weights = labels.size / (len(EMOTION_IDS) * counts)
    return weights[labels.astype(int)]


def _align_zero_based_proba(classes: Iterable[int], proba: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(EMOTION_IDS)), dtype=np.float64)
    for src_idx, label in enumerate(classes):
        label_idx = int(label)
        if 0 <= label_idx < len(EMOTION_IDS):
            aligned[:, label_idx] = proba[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze() <= 0
    aligned[missing, :] = 1.0 / len(EMOTION_IDS)
    aligned[~missing, :] = aligned[~missing, :] / row_sum[~missing]
    return aligned.astype(np.float32, copy=False)


@contextlib.contextmanager
def _time_limit(seconds: float) -> Iterable[None]:
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"candidate fit exceeded {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _fit_with_optional_weight(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    sample_weight = _sample_weight_balanced(y_train)
    if hasattr(model, "steps") and model.steps:
        final_step = model.steps[-1][0]
        try:
            return model.fit(x_train, y_train, **{f"{final_step}__sample_weight": sample_weight})
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return model.fit(x_train, y_train, model__sample_weight=sample_weight)
    except (KeyError, TypeError, ValueError):
        try:
            return model.fit(x_train, y_train, sample_weight=sample_weight)
        except (TypeError, ValueError):
            return model.fit(x_train, y_train)


def _predict_model_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not expose predict_proba")
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = getattr(model, "classes_", getattr(model[-1], "classes_", None) if hasattr(model, "__getitem__") else None)
    if classes is None:
        classes = np.arange(proba.shape[1])
    return _align_zero_based_proba(classes, proba)


def _build_tabular_model(name: str, *, seed: int, tabular_device: str) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)),
            ]
        )
    if name == "svc_rbf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        cache_size=2048,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=250,
                        max_leaf_nodes=31,
                        l2_regularization=0.02,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("lightgbm is not installed") from exc
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        objective="multiclass",
                        n_estimators=200,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=5,
                        min_child_samples=25,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is not installed") from exc
        xgb_device = "cuda" if tabular_device == "cuda" else "cpu"
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(EMOTION_IDS),
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        device=xgb_device,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("catboost is not installed") from exc
        task_type = "GPU" if tabular_device == "cuda" else "CPU"
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="MultiClass",
                        iterations=300,
                        depth=6,
                        learning_rate=0.04,
                        random_seed=seed,
                        task_type=task_type,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported tabular model: {name}")


def build_tabular_candidates(
    *,
    config: SearchConfig,
    examples: list[WaveformExample],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    tracker: StopTracker,
) -> list[Candidate]:
    if not config.include_tabular:
        return []
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        print(f"skip tabular candidates: scikit-learn unavailable: {exc}", flush=True)
        return []

    print("building strict window PPG feature matrices", flush=True)
    train_rows, y_train = _build_feature_rows(
        examples,
        train_idx,
        labels,
        window_seconds=config.window_seconds,
        windows_per_trial=config.train_windows_per_trial,
        rich=config.rich_features,
    )
    val_rows, _ = _build_feature_rows(
        examples,
        val_idx,
        labels,
        window_seconds=config.window_seconds,
        windows_per_trial=config.eval_windows_per_trial,
        rich=config.rich_features,
    )
    test_rows, _ = _build_feature_rows(
        examples,
        test_idx,
        labels,
        window_seconds=config.window_seconds,
        windows_per_trial=config.eval_windows_per_trial,
        rich=config.rich_features,
    )
    x_train, feature_names = _matrix_from_rows(train_rows)
    x_val, _ = _matrix_from_rows(val_rows, feature_names=feature_names)
    x_test, _ = _matrix_from_rows(test_rows, feature_names=feature_names)
    candidates: list[Candidate] = []
    for model_name in config.tabular_models:
        if tracker.stop_reason is not None:
            break
        start = time.time()
        try:
            model = _build_tabular_model(model_name, seed=config.seed, tabular_device=config.tabular_device)
            with _time_limit(config.tabular_model_timeout_seconds):
                model = _fit_with_optional_weight(model, x_train, y_train)
            val_probs = _predict_model_proba(model, x_val)
            val_metrics = _metrics_from_probs(labels[val_idx], val_probs)
            eligible, reason = _eligibility(val_metrics, config)
            candidate = Candidate(
                name=f"tabular:{model_name}",
                family="tabular",
                source="strict_window_ppg_features",
                val_probs=val_probs,
                val_metrics=val_metrics,
                eligible=eligible,
                eligibility_reason=reason,
                metadata={
                    "model_name": model_name,
                    "tabular_device": config.tabular_device,
                    "feature_count": len(feature_names),
                    "train_rows": int(x_train.shape[0]),
                    "eval_rows": int(x_val.shape[0]),
                    "train_windows_per_trial": config.train_windows_per_trial,
                    "eval_windows_per_trial": config.eval_windows_per_trial,
                    "rich_features": config.rich_features,
                    "elapsed_seconds": time.time() - start,
                },
                test_fn=lambda model=model, x_test=x_test: _predict_model_proba(model, x_test),
            )
        except Exception as exc:
            val_probs = np.full((val_idx.size, len(EMOTION_IDS)), 1.0 / len(EMOTION_IDS), dtype=np.float32)
            val_metrics = _metrics_from_probs(labels[val_idx], val_probs)
            candidate = Candidate(
                name=f"tabular:{model_name}",
                family="tabular",
                source="strict_window_ppg_features",
                val_probs=val_probs,
                val_metrics=val_metrics,
                eligible=False,
                eligibility_reason=f"fit_failed:{type(exc).__name__}:{exc}",
                metadata={
                    "model_name": model_name,
                    "tabular_device": config.tabular_device,
                    "feature_count": len(feature_names),
                    "train_rows": int(x_train.shape[0]),
                    "rich_features": config.rich_features,
                    "elapsed_seconds": time.time() - start,
                },
                test_fn=None,
            )
        candidates.append(candidate)
        tracker.observe(candidate)
        print(
            f"{candidate.name} eligible={candidate.eligible} reason={candidate.eligibility_reason} "
            f"val {metrics_summary_line(candidate.val_metrics)}",
            flush=True,
        )
    return candidates


def _ensemble_pool(candidates: list[Candidate], config: SearchConfig) -> list[Candidate]:
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        return []
    top1 = sorted(
        eligible,
        key=lambda c: (
            float(c.val_metrics["top1_accuracy"]),
            float(c.val_metrics["macro_f1"]),
            float(c.val_metrics["cohen_kappa"]),
        ),
        reverse=True,
    )
    top3 = sorted(
        eligible,
        key=lambda c: (
            float(c.val_metrics["top3_accuracy"]),
            float(c.val_metrics["macro_f1"]),
            float(c.val_metrics["top1_accuracy"]),
        ),
        reverse=True,
    )
    pool: list[Candidate] = []
    seen: set[str] = set()
    for candidate in itertools.chain(top1, top3):
        if candidate.name not in seen:
            pool.append(candidate)
            seen.add(candidate.name)
        if len(pool) >= config.max_ensemble_pool:
            break
    return pool


def build_ensemble_candidates(
    *,
    config: SearchConfig,
    candidates: list[Candidate],
    labels: np.ndarray,
    val_idx: np.ndarray,
    tracker: StopTracker,
) -> list[Candidate]:
    if not config.include_ensembles or tracker.stop_reason is not None:
        return []
    pool = _ensemble_pool(candidates, config)
    ensembles: list[Candidate] = []
    for size in range(2, max(2, config.max_ensemble_members) + 1):
        for members in itertools.combinations(pool, size):
            if tracker.stop_reason is not None:
                return ensembles
            member_names = tuple(member.name for member in members)
            val_probs = np.mean([member.val_probs for member in members], axis=0)
            val_metrics = _metrics_from_probs(labels[val_idx], val_probs)
            eligible, reason = _eligibility(val_metrics, config)
            name = f"ensemble:eq:{size}:" + "+".join(member.name.split(":", 1)[-1] for member in members)
            candidate = Candidate(
                name=name,
                family="ensemble",
                source="validation_probability_average",
                val_probs=val_probs,
                val_metrics=val_metrics,
                eligible=eligible,
                eligibility_reason=reason,
                metadata={
                    "members": member_names,
                    "weights": [1.0 / size for _ in members],
                    "selection_note": "equal-weight probability average selected by validation only",
                },
                test_fn=lambda members=members: np.mean([member.test_probs() for member in members], axis=0),
            )
            ensembles.append(candidate)
            tracker.observe(candidate)
            print(
                f"{candidate.name} eligible={candidate.eligible} reason={candidate.eligibility_reason} "
                f"val_top1={candidate.val_metrics['top1_accuracy']:.4f} "
                f"val_top3={candidate.val_metrics['top3_accuracy']:.4f} "
                f"val_macro_f1={candidate.val_metrics['macro_f1']:.4f}",
                flush=True,
            )
    return ensembles


def _rank_top1(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        [candidate for candidate in candidates if candidate.eligible],
        key=lambda c: (
            float(c.val_metrics["top1_accuracy"]),
            float(c.val_metrics["macro_f1"]),
            float(c.val_metrics["cohen_kappa"]),
        ),
        reverse=True,
    )


def _rank_top3(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        [candidate for candidate in candidates if candidate.eligible],
        key=lambda c: (
            float(c.val_metrics["top3_accuracy"]),
            float(c.val_metrics["macro_f1"]),
            float(c.val_metrics["top1_accuracy"]),
        ),
        reverse=True,
    )


def _leaderboard_rows(
    ranked: list[Candidate],
    *,
    labels: np.ndarray,
    test_idx: np.ndarray,
    selected_names: set[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked[:limit], start=1):
        row = candidate.row()
        row["rank"] = rank
        if candidate.name in selected_names:
            test_probs = candidate.test_probs()
            row["test_metrics"] = _metrics_from_probs(labels[test_idx], test_probs)
        rows.append(row)
    return rows


def _class_distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(int), minlength=len(EMOTION_IDS))
    return {EMOTION_ID_TO_LABEL[idx + 1]: int(counts[idx]) for idx in range(len(EMOTION_IDS))}


def run_search(config: SearchConfig) -> dict[str, Any]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device, device_info = select_torch_device(config.device)
    examples, dataset_summary = load_ppg_rating_examples(
        config.data_root,
        targets=RATING_TARGETS,
        max_subjects=config.max_subjects,
    )
    labels, subjects, _trials = _emotion_arrays(examples, config.label_column)
    split = make_subject_split(
        subjects,
        seed=config.split_seed,
        val_fraction=config.val_subject_fraction,
        test_fraction=config.test_subject_fraction,
    )
    print(
        f"device={device_info} examples={len(examples)} "
        f"split=train:{split.train.size}/val:{split.val.size}/test:{split.test.size}",
        flush=True,
    )

    dnn_candidates = load_dnn_candidates(
        config=config,
        examples=examples,
        labels=labels,
        split_val=split.val,
        split_test=split.test,
        device=device,
    )
    for candidate in dnn_candidates:
        print(
            f"{candidate.name} eligible={candidate.eligible} reason={candidate.eligibility_reason} "
            f"val {metrics_summary_line(candidate.val_metrics)}",
            flush=True,
        )

    tracker = StopTracker(config, initial_candidates=dnn_candidates)
    tabular_candidates = build_tabular_candidates(
        config=config,
        examples=examples,
        labels=labels,
        train_idx=split.train,
        val_idx=split.val,
        test_idx=split.test,
        device=device,
        tracker=tracker,
    )
    candidates = [*dnn_candidates, *tabular_candidates]
    ensemble_candidates = build_ensemble_candidates(
        config=config,
        candidates=candidates,
        labels=labels,
        val_idx=split.val,
        tracker=tracker,
    )
    candidates.extend(ensemble_candidates)

    top1_ranked = _rank_top1(candidates)
    top3_ranked = _rank_top3(candidates)
    selected_names: set[str] = set()
    if top1_ranked:
        selected_names.add(top1_ranked[0].name)
    if top3_ranked:
        selected_names.add(top3_ranked[0].name)
    selected_test_metrics: dict[str, Any] = {}
    for candidate in candidates:
        if candidate.name in selected_names:
            selected_test_metrics[candidate.name] = _metrics_from_probs(labels[split.test], candidate.test_probs())

    prior_val = _prior_baseline(labels[split.train], labels[split.val])
    result = {
        "task": "ppg_strict_streaming_emotion_classification_search",
        "target": config.label_column,
        "config": asdict(config),
        "protocol": {
            "ppg_only": True,
            "causal_window_seconds": config.window_seconds,
            "single_eval_window": config.eval_windows_per_trial == 1,
            "full_signal_or_trial_aggregation": False,
            "participant_grouped": True,
            "selection": "validation_only",
            "test_policy": "test metrics computed only for selected top-1/top-3 leaders in this script",
            "calibration": "none",
        },
        "dataset": asdict(dataset_summary),
        "split_counts": {
            "train_rows": int(split.train.size),
            "val_rows": int(split.val.size),
            "test_rows": int(split.test.size),
            "train_subjects": len(split.train_subjects),
            "val_subjects": len(split.val_subjects),
            "test_subjects": len(split.test_subjects),
        },
        "label_distribution": {
            "all": _class_distribution(labels),
            "train": _class_distribution(labels[split.train]),
            "val": _class_distribution(labels[split.val]),
            "test": _class_distribution(labels[split.test]),
        },
        "class_labels": list(EMOTION_IDS),
        "class_names": _labels_to_names(EMOTION_IDS),
        "device": device_info,
        "tabular_device": config.tabular_device,
        "prior_baseline_val_metrics": prior_val,
        "candidate_count": len(candidates),
        "dnn_candidate_count": len(dnn_candidates),
        "tabular_candidate_count": len(tabular_candidates),
        "ensemble_candidate_count": len(ensemble_candidates),
        "stop": tracker.state(),
        "candidates": [candidate.row() for candidate in candidates],
        "top1_leaderboard": _leaderboard_rows(
            top1_ranked,
            labels=labels,
            test_idx=split.test,
            selected_names=selected_names,
        ),
        "top3_leaderboard": _leaderboard_rows(
            top3_ranked,
            labels=labels,
            test_idx=split.test,
            selected_names=selected_names,
        ),
        "selected_test_metrics": selected_test_metrics,
        "git_commit": git_commit(),
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ppg_strict_emotion_search_result_{stamp}.json"
    result["result_path"] = str(out_path)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved result {out_path}", flush=True)
    if top1_ranked:
        print(f"top1 leader {top1_ranked[0].name} val {metrics_summary_line(top1_ranked[0].val_metrics)}", flush=True)
        if top1_ranked[0].name in selected_test_metrics:
            print(f"top1 selected test {metrics_summary_line(selected_test_metrics[top1_ranked[0].name])}", flush=True)
    if top3_ranked:
        print(f"top3 leader {top3_ranked[0].name} val {metrics_summary_line(top3_ranked[0].val_metrics)}", flush=True)
        if top3_ranked[0].name in selected_test_metrics:
            print(f"top3 selected test {metrics_summary_line(selected_test_metrics[top3_ranked[0].name])}", flush=True)
    print(f"stop {tracker.state()}", flush=True)
    return result


def _parse_model_list(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/extracted")
    parser.add_argument("--output-dir", default="artifacts/results")
    parser.add_argument("--label-column", choices=["ReportedType", "IntendedType"], default="ReportedType")
    parser.add_argument("--result-glob", default="ppg_*_reported_emotion_result_*.json")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--train-windows-per-trial", type=int, default=4)
    parser.add_argument("--eval-windows-per-trial", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--val-subject-fraction", type=float, default=0.15)
    parser.add_argument("--test-subject-fraction", type=float, default=0.15)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--tabular-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for XGBoost/CatBoost tabular candidates. CPU is default to avoid native GPU OOM aborts.",
    )
    parser.add_argument("--no-dnn", action="store_true")
    parser.add_argument("--no-tabular", action="store_true")
    parser.add_argument("--no-ensembles", action="store_true")
    parser.add_argument("--allow-amp-results", action="store_true")
    parser.add_argument(
        "--tabular-models",
        default="logistic_regression,svc_rbf,extra_trees,random_forest,hist_gradient_boosting,lightgbm,xgboost,catboost",
    )
    parser.add_argument("--basic-features", action="store_true")
    parser.add_argument("--max-ensemble-members", type=int, default=4)
    parser.add_argument("--max-ensemble-pool", type=int, default=10)
    parser.add_argument("--max-evaluated-candidates", type=int, default=60)
    parser.add_argument("--max-eligible-candidates", type=int, default=40)
    parser.add_argument("--no-improvement-patience", type=int, default=8)
    parser.add_argument("--improvement-delta", type=float, default=0.005)
    parser.add_argument("--max-improvements", type=int, default=5)
    parser.add_argument("--min-macro-f1", type=float, default=0.15)
    parser.add_argument("--max-prediction-share", type=float, default=0.75)
    parser.add_argument("--wall-clock-limit-seconds", type=float, default=4 * 60 * 60)
    parser.add_argument("--tabular-model-timeout-seconds", type=float, default=5 * 60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SearchConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        label_column=args.label_column,
        result_glob=args.result_glob,
        window_seconds=args.window_seconds,
        train_windows_per_trial=args.train_windows_per_trial,
        eval_windows_per_trial=args.eval_windows_per_trial,
        split_seed=args.split_seed,
        seed=args.seed,
        val_subject_fraction=args.val_subject_fraction,
        test_subject_fraction=args.test_subject_fraction,
        max_subjects=args.max_subjects,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        tabular_device=args.tabular_device,
        include_dnn=not args.no_dnn,
        include_tabular=not args.no_tabular,
        include_ensembles=not args.no_ensembles,
        allow_amp_results=args.allow_amp_results,
        tabular_models=_parse_model_list(args.tabular_models),
        rich_features=not args.basic_features,
        max_ensemble_members=args.max_ensemble_members,
        max_ensemble_pool=args.max_ensemble_pool,
        max_evaluated_candidates=args.max_evaluated_candidates,
        max_eligible_candidates=args.max_eligible_candidates,
        no_improvement_patience=args.no_improvement_patience,
        improvement_delta=args.improvement_delta,
        max_improvements=args.max_improvements,
        min_macro_f1=args.min_macro_f1,
        max_prediction_share=args.max_prediction_share,
        wall_clock_limit_seconds=args.wall_clock_limit_seconds,
        tabular_model_timeout_seconds=args.tabular_model_timeout_seconds,
    )
    run_search(config)


if __name__ == "__main__":
    main()
