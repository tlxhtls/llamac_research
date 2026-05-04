from __future__ import annotations

import numpy as np

from llamac_research.metrics import compute_classification_metrics


def test_required_classification_metrics() -> None:
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 2, 4, 5]
    score = np.eye(5)[np.array(y_pred) - 1]
    metrics = compute_classification_metrics(y_true, y_pred, score).to_dict()
    for key in [
        "top1_accuracy",
        "top2_accuracy",
        "top3_accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "cohen_kappa",
        "confusion_matrix",
    ]:
        assert key in metrics
    assert metrics["top1_accuracy"] == 0.8
    assert len(metrics["confusion_matrix"]) == 5
