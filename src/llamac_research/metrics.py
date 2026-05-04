"""Metric helpers for five-class LLaMAC emotion classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .labels import EMOTION_ID_TO_LABEL, EMOTION_IDS, EMOTION_LABELS


@dataclass(frozen=True)
class ClassificationMetrics:
    """Serializable metric bundle."""

    top1_accuracy: float
    top2_accuracy: float
    top3_accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    cohen_kappa: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top1_accuracy": self.top1_accuracy,
            "top2_accuracy": self.top2_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "cohen_kappa": self.cohen_kappa,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
        }


def _top_k_accuracy(y_true: np.ndarray, y_score: np.ndarray, labels: Sequence[int], k: int) -> float:
    if y_true.size == 0:
        return float("nan")
    label_arr = np.asarray(labels)
    k = min(k, y_score.shape[1])
    top_idx = np.argpartition(y_score, kth=y_score.shape[1] - k, axis=1)[:, -k:]
    top_labels = label_arr[top_idx]
    return float(np.mean([yt in row for yt, row in zip(y_true, top_labels, strict=False)]))


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    labels: Sequence[int] = EMOTION_IDS,
    label_names: Sequence[str] = EMOTION_LABELS,
) -> ClassificationMetrics:
    """Compute the required LLaMAC classification metrics."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    label_list = list(labels)
    if y_score is None:
        score = np.zeros((y_pred_arr.size, len(label_list)), dtype=float)
        index = {label: idx for idx, label in enumerate(label_list)}
        for row, pred in enumerate(y_pred_arr):
            if pred in index:
                score[row, index[pred]] = 1.0
    else:
        score = np.asarray(y_score, dtype=float)
    top1 = float(np.mean(y_true_arr == y_pred_arr)) if y_true_arr.size else float("nan")
    top2 = _top_k_accuracy(y_true_arr, score, label_list, 2)
    top3 = _top_k_accuracy(y_true_arr, score, label_list, 3)
    macro = float(f1_score(y_true_arr, y_pred_arr, labels=label_list, average="macro", zero_division=0))
    weighted = float(f1_score(y_true_arr, y_pred_arr, labels=label_list, average="weighted", zero_division=0))
    balanced = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr, labels=label_list))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_list)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=label_list,
        zero_division=0,
    )
    per_class = {
        name: {
            "label_id": int(label),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, name, p, r, f, s in zip(label_list, label_names, precision, recall, f1, support, strict=True)
    }
    return ClassificationMetrics(
        top1_accuracy=top1,
        top2_accuracy=top2,
        top3_accuracy=top3,
        macro_f1=macro,
        weighted_f1=weighted,
        balanced_accuracy=balanced,
        cohen_kappa=kappa,
        confusion_matrix=cm.astype(int).tolist(),
        per_class=per_class,
    )


def align_proba_columns(model_classes: Sequence[int], proba: np.ndarray, labels: Sequence[int] = EMOTION_IDS) -> np.ndarray:
    """Align estimator predict_proba columns to the canonical label order."""
    classes = [int(c) for c in model_classes]
    label_list = list(labels)
    aligned = np.zeros((proba.shape[0], len(label_list)), dtype=float)
    for src_idx, label in enumerate(classes):
        if label in label_list:
            aligned[:, label_list.index(label)] = proba[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze() == 0
    if np.any(missing):
        aligned[missing, :] = 1.0 / len(label_list)
    else:
        aligned = aligned / row_sum
    return aligned


def metrics_summary_line(metrics: dict[str, Any]) -> str:
    """Compact human-readable metric line for CLI logs."""
    return (
        f"top1={metrics['top1_accuracy']:.4f} top2={metrics['top2_accuracy']:.4f} "
        f"top3={metrics['top3_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"balanced_acc={metrics['balanced_accuracy']:.4f} kappa={metrics['cohen_kappa']:.4f}"
    )
