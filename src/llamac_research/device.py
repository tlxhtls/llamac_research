"""Device/backend selection helpers.

Public CLIs remain CPU-compatible while recording whether CUDA/GPU acceleration was
requested, available, and actually selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

DeviceRequest = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class BackendSelection:
    """Resolved backend metadata for reproducible experiment logs."""

    requested: str
    selected: str
    backend: str
    gpu_name: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def select_torch_device(requested: DeviceRequest = "auto") -> BackendSelection:
    """Resolve a PyTorch-style device without requiring torch at import time."""
    if requested == "cpu":
        return BackendSelection(requested=requested, selected="cpu", backend="torch")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional install
        if requested == "cuda":
            return BackendSelection(
                requested=requested,
                selected="cpu",
                backend="torch",
                fallback_reason=f"torch import failed: {exc}",
            )
        return BackendSelection(
            requested=requested,
            selected="cpu",
            backend="torch",
            fallback_reason="torch is not installed",
        )

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return BackendSelection(
            requested=requested,
            selected="cuda",
            backend="torch",
            gpu_name=torch.cuda.get_device_name(idx),
        )
    return BackendSelection(
        requested=requested,
        selected="cpu",
        backend="torch",
        fallback_reason="torch.cuda.is_available() is false",
    )


def _probe_lightgbm_device(device_type: str) -> tuple[bool, str | None]:
    """Return whether this LightGBM build can train with a requested device_type."""
    try:
        import lightgbm as lgb
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, f"lightgbm import failed: {exc}"

    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.8, 0.9],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    params = {
        "objective": "multiclass",
        "num_class": 2,
        "metric": "multi_logloss",
        "verbosity": -1,
        "device_type": device_type,
        "num_threads": 1,
        "force_col_wise": True,
    }
    try:
        lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=1)
    except Exception as exc:  # pragma: no cover - depends on local build
        return False, str(exc).splitlines()[0]
    return True, None


def select_lightgbm_device(requested: DeviceRequest = "auto") -> BackendSelection:
    """Resolve LightGBM device_type with CUDA/GPU probing and CPU fallback."""
    if requested == "cpu":
        return BackendSelection(requested=requested, selected="cpu", backend="lightgbm")

    failures: list[str] = []
    for candidate in ("cuda", "gpu"):
        ok, reason = _probe_lightgbm_device(candidate)
        if ok:
            return BackendSelection(requested=requested, selected=candidate, backend="lightgbm")
        failures.append(f"{candidate}: {reason}")

    return BackendSelection(
        requested=requested,
        selected="cpu",
        backend="lightgbm",
        fallback_reason="; ".join(failures),
    )


def select_xgboost_device(requested: DeviceRequest = "auto") -> BackendSelection:
    """Resolve XGBoost device string. XGBoost itself will still validate at fit time."""
    if requested == "cpu":
        return BackendSelection(requested=requested, selected="cpu", backend="xgboost")
    torch_selection = select_torch_device(requested="auto")
    if torch_selection.selected == "cuda":
        return BackendSelection(
            requested=requested,
            selected="cuda",
            backend="xgboost",
            gpu_name=torch_selection.gpu_name,
        )
    return BackendSelection(
        requested=requested,
        selected="cpu",
        backend="xgboost",
        fallback_reason=torch_selection.fallback_reason or "CUDA was not detected",
    )
