"""Raw-waveform DNN training utilities for LLaMAC PPG rating prediction."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import polars as pl
import torch
from torch import nn

from . import __version__
from .features import _read_csv, discover_subject_dirs, extract_trial_number, natural_key
from .labels import add_target_columns, filter_official_valid_trials

RATING_TARGETS: tuple[str, ...] = ("Valence", "Arousal", "Dominance", "Liking")
RATING_TARGET_RANGES: dict[str, float] = {
    "Valence": 4.0,
    "Arousal": 4.0,
    "Dominance": 4.0,
    "Liking": 2.0,
}
RATING_TARGET_MINS: dict[str, float] = {
    "Valence": 1.0,
    "Arousal": 1.0,
    "Dominance": 1.0,
    "Liking": 1.0,
}


@dataclass(frozen=True)
class WaveformExample:
    """One trial-level PPG waveform and its rating targets."""

    subject_id: str
    trial: int
    ppg: np.ndarray
    ppg_time: np.ndarray | None
    targets: dict[str, float]
    labels: dict[str, int]


@dataclass(frozen=True)
class WaveformDatasetSummary:
    """Shape and coverage summary for loaded raw PPG trials."""

    examples: int
    participants: int
    targets: tuple[str, ...]
    length_min: int
    length_median: float
    length_max: int


@dataclass(frozen=True)
class TrainSplit:
    """Subject-grouped train/validation/test indices."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


@dataclass(frozen=True)
class RatingCnnConfig:
    """Configuration for raw PPG 1D-CNN rating prediction."""

    data_root: str = "data/extracted"
    output_dir: str = "artifacts/results"
    targets: tuple[str, ...] = RATING_TARGETS
    target_transform: Literal["standard", "range", "raw"] = "standard"
    model_arch: Literal[
        "cnn",
        "cnn_gru",
        "cnn_lstm",
        "cnn_multihead",
        "cnn_attention_multihead",
        "cnn_derivative_multihead",
        "cnn_emotion_multihead",
        "cnn_ordinal_multihead",
        "cnn_statfusion_multihead",
        "cnn_transformer_multihead",
        "idcnn_multihead",
        "tcn_multihead",
        "resnet_multihead",
    ] = "cnn"
    target_length: int = 1920
    min_window_seconds: float = 5.0
    max_window_seconds: float = 30.0
    eval_window_seconds: float = 30.0
    train_window_anchor: Literal["random", "first", "center", "last", "even"] = "random"
    input_normalization: Literal["robust", "zscore", "none"] = "robust"
    input_clip_value: float | None = None
    output_activation: Literal["linear", "sigmoid_range"] = "linear"
    input_noise_std: float = 0.0
    time_mask_fraction: float = 0.0
    official_valid_only: bool = False
    train_windows_per_trial: int = 1
    seed: int = 42
    split_seed: int | None = None
    batch_size: int = 256
    epochs: int = 80
    patience: int = 14
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    shared_dropout: float = 0.25
    head_dropout: float = 0.20
    eval_windows_per_trial: int = 1
    loss_name: Literal[
        "smooth_l1_scaled",
        "mae_scaled",
        "mae_raw",
        "range_mae_raw",
        "range_mae_var_raw",
        "range_mae_under_var_raw",
        "range_epsilon_mae_raw",
        "range_epsilon_mae_var_raw",
        "smooth_l1_raw",
        "range_smooth_l1_raw",
    ] = "smooth_l1_scaled"
    ordinal_aux_weight: float = 0.0
    emotion_aux_weight: float = 0.0
    emotion_label_column: Literal["ReportedType", "IntendedType"] = "ReportedType"
    mixup_alpha: float = 0.0
    ema_decay: float = 0.0
    variance_loss_weight: float = 0.02
    variance_target_weights: tuple[float, ...] | None = None
    correlation_loss_weight: float = 0.0
    correlation_target_weights: tuple[float, ...] | None = None
    target_weights: tuple[float, ...] | None = None
    refit_train_val: bool = False
    val_subject_fraction: float = 0.15
    test_subject_fraction: float = 0.15
    max_subjects: int | None = None
    num_workers: int = 4
    device: Literal["auto", "cuda", "cpu"] = "auto"
    amp: bool = True


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _as_float_array(series: pl.Series) -> np.ndarray:
    return series.cast(pl.Float64, strict=False).to_numpy().astype(np.float32, copy=False)


def load_ppg_rating_examples(
    data_root: str | Path,
    *,
    targets: Sequence[str] = RATING_TARGETS,
    max_subjects: int | None = None,
    official_valid_only: bool = False,
) -> tuple[list[WaveformExample], WaveformDatasetSummary]:
    """Load raw PPG trial waveforms and target ratings from extracted LLaMAC folders."""

    root = Path(data_root)
    examples: list[WaveformExample] = []
    lengths: list[int] = []
    subject_dirs = discover_subject_dirs(root, limit_subjects=max_subjects)
    required_targets = set(targets)
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        answer_path = subject_dir / "answer.csv"
        answer = add_target_columns(_read_csv(answer_path))
        if "SubjectID" not in answer.columns:
            answer = answer.with_columns(pl.lit(int(subject_id)).alias("SubjectID"))
        if official_valid_only:
            answer = filter_official_valid_trials(answer)
        missing = required_targets.difference(answer.columns)
        if missing:
            raise ValueError(f"{answer_path} missing rating targets: {sorted(missing)}")
        answer_by_trial: dict[int, dict[str, Any]] = {}
        for row in answer.iter_rows(named=True):
            try:
                trial = int(row["Trial"])
            except (TypeError, ValueError):
                continue
            answer_by_trial[trial] = row

        for band_path in sorted(subject_dir.glob("band_*.csv"), key=natural_key):
            trial = extract_trial_number(band_path)
            if trial is None or trial not in answer_by_trial:
                continue
            band = _read_csv(band_path)
            if "PPG" not in band.columns:
                continue
            ppg = _as_float_array(band["PPG"])
            if ppg.size < 16 or not np.isfinite(ppg).any():
                continue
            row = answer_by_trial[trial]
            target_values: dict[str, float] = {}
            skip = False
            for target in targets:
                try:
                    value = float(row[target])
                except (TypeError, ValueError):
                    skip = True
                    break
                if not math.isfinite(value):
                    skip = True
                    break
                target_values[target] = value
            if skip:
                continue
            ppg_time = _as_float_array(band["PPG_Time"]) if "PPG_Time" in band.columns else None
            examples.append(
                WaveformExample(
                    subject_id=str(subject_id),
                    trial=int(trial),
                    ppg=ppg,
                    ppg_time=ppg_time,
                    targets=target_values,
                    labels={
                        "ReportedType": int(row["ReportedType"]),
                        "IntendedType": int(row["IntendedType"]),
                    },
                )
            )
            lengths.append(int(ppg.size))

    if not examples:
        raise FileNotFoundError(f"No PPG rating examples found under {root}")
    length_arr = np.asarray(lengths, dtype=float)
    summary = WaveformDatasetSummary(
        examples=len(examples),
        participants=len({ex.subject_id for ex in examples}),
        targets=tuple(targets),
        length_min=int(np.min(length_arr)),
        length_median=float(np.median(length_arr)),
        length_max=int(np.max(length_arr)),
    )
    return examples, summary


def resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    """Resample one irregular-length signal to a fixed number of samples."""

    y = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(y)
    if not finite.any():
        return np.zeros(target_length, dtype=np.float32)
    if not finite.all():
        idx = np.arange(y.size)
        y = y.copy()
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    if y.size == target_length:
        return y.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, num=y.size, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(new_x, old_x, y).astype(np.float32, copy=False)


def window_to_model_input(
    values: np.ndarray,
    *,
    window_seconds: float,
    max_window_seconds: float,
    target_length: int,
    input_normalization: Literal["robust", "zscore", "none"] = "robust",
    input_clip_value: float | None = None,
) -> np.ndarray:
    """Convert a PPG window to ONNX-friendly [2, target_length] input."""

    signal = resample_1d(values, target_length)
    if input_normalization == "robust":
        # Per-window robust normalization keeps subject-specific amplitude from dominating.
        center = float(np.median(signal))
        q25, q75 = np.quantile(signal, [0.25, 0.75])
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(signal))
        if not math.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = (signal - center) / scale
    elif input_normalization == "zscore":
        center = float(np.mean(signal))
        scale = float(np.std(signal))
        if not math.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        normalized = (signal - center) / scale
    elif input_normalization == "none":
        normalized = signal
    else:
        raise ValueError(f"Unsupported input_normalization={input_normalization!r}")
    if input_clip_value is not None and input_clip_value > 0:
        clip = float(input_clip_value)
        normalized = np.clip(normalized, -clip, clip)
    duration_ratio = np.full(target_length, min(float(window_seconds) / max_window_seconds, 1.0), dtype=np.float32)
    return np.stack([normalized.astype(np.float32, copy=False), duration_ratio], axis=0)


def crop_ppg_window(
    example: WaveformExample,
    *,
    window_seconds: float,
    rng: np.random.Generator | None = None,
    anchor: Literal["random", "first", "center", "last"] = "center",
    start_fraction: float | None = None,
) -> np.ndarray:
    """Crop a PPG window by time when possible, falling back to sample-count estimates."""

    y = example.ppg
    if y.size == 0:
        return np.zeros(1, dtype=np.float32)
    if example.ppg_time is not None and example.ppg_time.size == y.size:
        t = example.ppg_time.astype(np.float64, copy=False)
        finite = np.isfinite(t)
        if finite.sum() >= 2:
            start_t = float(np.nanmin(t))
            end_t = float(np.nanmax(t))
            available = max(0.0, end_t - start_t)
            duration = min(float(window_seconds), available) if available > 0 else float(window_seconds)
            if start_fraction is not None and available > duration:
                fraction = min(max(float(start_fraction), 0.0), 1.0)
                lo = start_t + (available - duration) * fraction
            elif anchor == "random" and rng is not None and available > duration:
                lo = float(rng.uniform(start_t, end_t - duration))
            elif anchor == "first":
                lo = start_t
            elif anchor == "last":
                lo = end_t - duration
            else:
                lo = start_t + max(0.0, available - duration) / 2.0
            hi = lo + duration
            mask = (t >= lo) & (t <= hi)
            if mask.sum() >= 16:
                return y[mask].astype(np.float32, copy=False)

    # Fallback: infer a 60 Hz-ish sample window from count.
    n = y.size
    samples = max(16, min(n, int(round(n * min(float(window_seconds), 60.0) / 60.0))))
    if start_fraction is not None and n > samples:
        start = int(round((n - samples) * min(max(float(start_fraction), 0.0), 1.0)))
    elif anchor == "random" and rng is not None and n > samples:
        start = int(rng.integers(0, n - samples + 1))
    elif anchor == "first":
        start = 0
    elif anchor == "last":
        start = max(0, n - samples)
    else:
        start = max(0, (n - samples) // 2)
    return y[start : start + samples].astype(np.float32, copy=False)


def build_target_arrays(
    examples: Sequence[WaveformExample],
    *,
    targets: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y[N, targets], subjects[N], trials[N]."""

    y = np.empty((len(examples), len(targets)), dtype=np.float32)
    subjects = np.empty(len(examples), dtype=object)
    trials = np.empty(len(examples), dtype=np.int64)
    for idx, example in enumerate(examples):
        y[idx] = [float(example.targets[target]) for target in targets]
        subjects[idx] = example.subject_id
        trials[idx] = example.trial
    return y, subjects, trials


class PpgWindowDataset(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """Dataset that samples streaming-style windows from each trial."""

    def __init__(
        self,
        examples: Sequence[WaveformExample],
        indices: np.ndarray,
        y_scaled: np.ndarray,
        *,
        target_length: int,
        min_window_seconds: float,
        max_window_seconds: float,
        eval_window_seconds: float,
        train_window_anchor: Literal["random", "first", "center", "last", "even"] = "random",
        input_normalization: Literal["robust", "zscore", "none"] = "robust",
        input_clip_value: float | None = None,
        input_noise_std: float = 0.0,
        time_mask_fraction: float = 0.0,
        windows_per_trial: int = 1,
        train: bool,
        seed: int,
        emotion_label_column: str | None = None,
    ) -> None:
        self.examples = examples
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y_scaled = y_scaled.astype(np.float32, copy=False)
        self.target_length = int(target_length)
        self.min_window_seconds = float(min_window_seconds)
        self.max_window_seconds = float(max_window_seconds)
        self.eval_window_seconds = float(eval_window_seconds)
        self.train_window_anchor = train_window_anchor
        self.input_normalization = input_normalization
        self.input_clip_value = input_clip_value
        self.input_noise_std = float(input_noise_std)
        self.time_mask_fraction = float(time_mask_fraction)
        self.windows_per_trial = max(1, int(windows_per_trial))
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)
        self.emotion_label_column = emotion_label_column

    def __len__(self) -> int:
        return int(self.indices.size * self.windows_per_trial)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        base_count = self.indices.size
        idx = int(self.indices[item % base_count])
        repeat_idx = int(item // base_count)
        example = self.examples[idx]
        if self.train:
            seconds = float(self.rng.uniform(self.min_window_seconds, self.max_window_seconds))
            if self.train_window_anchor == "random":
                anchor: Literal["random", "first", "center", "last"] = "random"
                rng = self.rng
                start_fraction = None
            elif self.train_window_anchor == "even":
                anchor = "center"
                rng = None
                start_fraction = repeat_idx / max(1, self.windows_per_trial - 1)
            else:
                anchor = self.train_window_anchor
                rng = None
                start_fraction = None
        else:
            seconds = self.eval_window_seconds
            if self.windows_per_trial == 1:
                start_fraction = 1.0
            else:
                start_fraction = repeat_idx / max(1, self.windows_per_trial - 1)
            anchor = "center"
            rng = None
        window = crop_ppg_window(
            example,
            window_seconds=seconds,
            rng=rng,
            anchor=anchor,
            start_fraction=start_fraction,
        )
        x = window_to_model_input(
            window,
            window_seconds=seconds,
            max_window_seconds=self.max_window_seconds,
            target_length=self.target_length,
            input_normalization=self.input_normalization,
            input_clip_value=self.input_clip_value,
        )
        if self.train:
            if self.input_noise_std > 0:
                x[0] = x[0] + self.rng.normal(0.0, self.input_noise_std, size=x[0].shape).astype(np.float32)
            if self.time_mask_fraction > 0:
                width = int(round(self.target_length * min(max(self.time_mask_fraction, 0.0), 0.5)))
                if width > 0 and width < self.target_length:
                    start = int(self.rng.integers(0, self.target_length - width + 1))
                    fill = float(np.median(x[0]))
                    x[0, start : start + width] = fill
        output: tuple[torch.Tensor, ...] = (torch.from_numpy(x), torch.from_numpy(self.y_scaled[idx]))
        if self.emotion_label_column is not None:
            # LLaMAC emotion ids are 1..5; cross_entropy expects 0..4.
            label = int(example.labels[self.emotion_label_column]) - 1
            output = (*output, torch.tensor(label, dtype=torch.long))
        return output


def make_subject_split(
    subjects: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> TrainSplit:
    """Create participant-disjoint train/validation/test splits."""

    unique_subjects = np.array(sorted({str(s) for s in subjects}, key=natural_key), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    n_subjects = unique_subjects.size
    n_test = max(1, int(round(n_subjects * test_fraction)))
    n_val = max(1, int(round(n_subjects * val_fraction)))
    if n_test + n_val >= n_subjects:
        raise ValueError("Not enough subjects for requested validation/test fractions")
    test_subjects = unique_subjects[:n_test]
    val_subjects = unique_subjects[n_test : n_test + n_val]
    train_subjects = unique_subjects[n_test + n_val :]
    train = np.flatnonzero(np.isin(subjects.astype(str), train_subjects.astype(str)))
    val = np.flatnonzero(np.isin(subjects.astype(str), val_subjects.astype(str)))
    test = np.flatnonzero(np.isin(subjects.astype(str), test_subjects.astype(str)))
    return TrainSplit(
        train=train,
        val=val,
        test=test,
        train_subjects=tuple(str(s) for s in train_subjects),
        val_subjects=tuple(str(s) for s in val_subjects),
        test_subjects=tuple(str(s) for s in test_subjects),
    )


def standardize_targets(
    y: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y[train_idx].mean(axis=0)
    std = y[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    return (y - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def scale_rating_targets(
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    targets: Sequence[str],
    transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale regression targets while preserving the inverse raw-rating mapping."""

    if transform == "standard":
        return standardize_targets(y, train_idx)
    if transform == "range":
        mean = np.asarray([RATING_TARGET_MINS[target] for target in targets], dtype=np.float32)
        std = np.asarray([RATING_TARGET_RANGES[target] for target in targets], dtype=np.float32)
        return ((y - mean) / std).astype(np.float32), mean, std
    if transform == "raw":
        mean = np.zeros(len(targets), dtype=np.float32)
        std = np.ones(len(targets), dtype=np.float32)
        return y.astype(np.float32), mean, std
    raise ValueError(f"Unsupported target_transform={transform!r}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: Sequence[str]) -> dict[str, Any]:
    """Compute MAE/RMSE/R2 per target and averaged across targets."""

    out: dict[str, Any] = {}
    maes: list[float] = []
    normalized_maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    for idx, target in enumerate(targets):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        err = pred - true
        mae = float(np.mean(np.abs(err)))
        target_range = float(RATING_TARGET_RANGES.get(target, np.nanmax(true) - np.nanmin(true)))
        normalized_mae = float(mae / target_range) if target_range > 0 else math.nan
        rmse = float(np.sqrt(np.mean(err**2)))
        denom = float(np.sum((true - np.mean(true)) ** 2))
        r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else math.nan
        out[target] = {"mae": mae, "range_normalized_mae": normalized_mae, "rmse": rmse, "r2": r2}
        maes.append(mae)
        if math.isfinite(normalized_mae):
            normalized_maes.append(normalized_mae)
        rmses.append(rmse)
        if math.isfinite(r2):
            r2s.append(r2)
    out["macro_mae"] = float(np.mean(maes))
    out["macro_range_normalized_mae"] = float(np.mean(normalized_maes)) if normalized_maes else math.nan
    out["macro_rmse"] = float(np.mean(rmses))
    out["macro_r2"] = float(np.mean(r2s)) if r2s else math.nan
    return out


def mean_baseline_metrics(y_train: np.ndarray, y_eval: np.ndarray, targets: Sequence[str]) -> dict[str, Any]:
    """Evaluate the train-target-mean baseline on an evaluation split."""

    pred = np.broadcast_to(y_train.mean(axis=0), y_eval.shape).astype(np.float32, copy=False)
    return regression_metrics(y_eval, pred, targets)


def _target_min_range_arrays(targets: Sequence[str], *, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    mins = torch.as_tensor([RATING_TARGET_MINS[target] for target in targets], dtype=torch.float32, device=device).view(1, -1)
    ranges = torch.as_tensor([RATING_TARGET_RANGES[target] for target in targets], dtype=torch.float32, device=device).view(1, -1)
    return mins, ranges


def apply_output_activation_torch(
    pred_scaled: torch.Tensor,
    *,
    output_activation: str,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    targets: Sequence[str],
) -> torch.Tensor:
    if output_activation == "linear":
        return pred_scaled
    if output_activation == "sigmoid_range":
        target_mins, target_ranges = _target_min_range_arrays(targets, device=pred_scaled.device)
        pred_raw = target_mins + target_ranges * torch.sigmoid(pred_scaled)
        return (pred_raw - target_mean) / target_std
    raise ValueError(f"Unsupported output_activation={output_activation!r}")


def apply_output_activation_numpy(
    pred_scaled: np.ndarray,
    *,
    output_activation: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    targets: Sequence[str],
) -> np.ndarray:
    if output_activation == "linear":
        return pred_scaled
    if output_activation == "sigmoid_range":
        target_mins = np.asarray([RATING_TARGET_MINS[target] for target in targets], dtype=np.float32).reshape(1, -1)
        target_ranges = np.asarray([RATING_TARGET_RANGES[target] for target in targets], dtype=np.float32).reshape(1, -1)
        pred_raw = target_mins + target_ranges / (1.0 + np.exp(-pred_scaled))
        return ((pred_raw - target_mean.reshape(1, -1)) / target_std.reshape(1, -1)).astype(np.float32)
    raise ValueError(f"Unsupported output_activation={output_activation!r}")


def select_torch_device(requested: str) -> tuple[str, dict[str, Any]]:
    import torch

    info: dict[str, Any] = {"requested": requested}
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "selected": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_gib": props.total_memory / 1024**3,
                "cuda_version": torch.version.cuda,
            }
        )
        return device, info
    if requested == "cuda":
        info.update({"selected": "cpu", "fallback_reason": "torch.cuda.is_available() is false"})
    else:
        info.update({"selected": "cpu"})
    return "cpu", info


def train_rating_cnn(config: RatingCnnConfig) -> dict[str, Any]:
    """Train a compact 1D-CNN on raw PPG and save the best validation checkpoint."""

    try:
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover - depends on optional dnn group
        raise RuntimeError("PyTorch is required. Install with `uv sync --group dnn`.") from exc

    set_reproducible_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device, device_info = select_torch_device(config.device)
    examples, dataset_summary = load_ppg_rating_examples(
        config.data_root,
        targets=config.targets,
        max_subjects=config.max_subjects,
        official_valid_only=config.official_valid_only,
    )
    y, subjects, trials = build_target_arrays(examples, targets=config.targets)
    split = make_subject_split(
        subjects,
        seed=config.split_seed if config.split_seed is not None else config.seed,
        val_fraction=config.val_subject_fraction,
        test_fraction=config.test_subject_fraction,
    )
    y_scaled, y_mean, y_std = scale_rating_targets(
        y,
        split.train,
        targets=config.targets,
        transform=config.target_transform,
    )
    emotion_label_column = config.emotion_label_column if config.emotion_aux_weight > 0 else None

    train_ds = PpgWindowDataset(
        examples,
        split.train,
        y_scaled,
        target_length=config.target_length,
        min_window_seconds=config.min_window_seconds,
        max_window_seconds=config.max_window_seconds,
        eval_window_seconds=config.eval_window_seconds,
        train_window_anchor=config.train_window_anchor,
        input_normalization=config.input_normalization,
        input_clip_value=config.input_clip_value,
        input_noise_std=config.input_noise_std,
        time_mask_fraction=config.time_mask_fraction,
        windows_per_trial=config.train_windows_per_trial,
        train=True,
        seed=config.seed,
        emotion_label_column=emotion_label_column,
    )
    val_ds = PpgWindowDataset(
        examples,
        split.val,
        y_scaled,
        target_length=config.target_length,
        min_window_seconds=config.min_window_seconds,
        max_window_seconds=config.max_window_seconds,
        eval_window_seconds=config.eval_window_seconds,
        input_normalization=config.input_normalization,
        input_clip_value=config.input_clip_value,
        windows_per_trial=config.eval_windows_per_trial,
        train=False,
        seed=config.seed,
        emotion_label_column=emotion_label_column,
    )
    test_ds = PpgWindowDataset(
        examples,
        split.test,
        y_scaled,
        target_length=config.target_length,
        min_window_seconds=config.min_window_seconds,
        max_window_seconds=config.max_window_seconds,
        eval_window_seconds=config.eval_window_seconds,
        input_normalization=config.input_normalization,
        input_clip_value=config.input_clip_value,
        windows_per_trial=config.eval_windows_per_trial,
        train=False,
        seed=config.seed,
        emotion_label_column=emotion_label_column,
    )
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    model = create_rating_model(
        config.model_arch,
        n_outputs=len(config.targets),
        shared_dropout=config.shared_dropout,
        head_dropout=config.head_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device == "cuda")
    ema_state = clone_state_dict(model.state_dict()) if config.ema_decay > 0 else None

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = output_dir / f"ppg_{config.model_arch}_ratings_best_{stamp}.pt"

    best_val_metric = math.inf
    best_val_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device=device,
            loss_name=config.loss_name,
            output_activation=config.output_activation,
            ordinal_aux_weight=config.ordinal_aux_weight,
            emotion_aux_weight=config.emotion_aux_weight,
            mixup_alpha=config.mixup_alpha,
            target_mean=y_mean,
            target_std=y_std,
            targets=config.targets,
            target_weights=config.target_weights,
            variance_loss_weight=config.variance_loss_weight,
            variance_target_weights=config.variance_target_weights,
            correlation_loss_weight=config.correlation_loss_weight,
            correlation_target_weights=config.correlation_target_weights,
            optimizer=optimizer,
            scaler=scaler,
            ema_state=ema_state,
            ema_decay=config.ema_decay,
        )
        raw_state = None
        if ema_state is not None:
            raw_state = clone_state_dict(model.state_dict())
            model.load_state_dict(ema_state)
        val_loss = _run_epoch(
            model,
            val_loader,
            device=device,
            loss_name=config.loss_name,
            output_activation=config.output_activation,
            ordinal_aux_weight=config.ordinal_aux_weight,
            emotion_aux_weight=config.emotion_aux_weight,
            mixup_alpha=0.0,
            target_mean=y_mean,
            target_std=y_std,
            targets=config.targets,
            target_weights=config.target_weights,
            variance_loss_weight=config.variance_loss_weight,
            variance_target_weights=config.variance_target_weights,
            correlation_loss_weight=config.correlation_loss_weight,
            correlation_target_weights=config.correlation_target_weights,
        )
        val_pred_scaled = _predict_averaged(
            model,
            val_loader,
            device=device,
            output_activation=config.output_activation,
            target_mean=y_mean,
            target_std=y_std,
            targets=config.targets,
        )
        val_pred = val_pred_scaled * y_std + y_mean
        val_true = y[split.val]
        val_metrics = regression_metrics(val_true, val_pred, config.targets)
        val_macro_mae = float(val_metrics["macro_mae"])
        val_norm_mae = float(val_metrics["macro_range_normalized_mae"])
        if raw_state is not None:
            model.load_state_dict(raw_state)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_mae": val_macro_mae,
            "val_macro_range_normalized_mae": val_norm_mae,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            " ".join(
                [
                    f"epoch={epoch:03d}",
                    f"train_loss={train_loss:.4f}",
                    f"val_loss={val_loss:.4f}",
                    f"val_mae={val_macro_mae:.4f}",
                    f"val_norm_mae={val_norm_mae:.4f}",
                    f"lr={row['lr']:.2e}",
                ]
            ),
            flush=True,
        )
        if val_macro_mae < best_val_metric - 1e-5:
            best_val_metric = val_macro_mae
            best_val_loss = float(val_loss)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": clone_state_dict(ema_state if ema_state is not None else model.state_dict(), device="cpu"),
                    "config": asdict(config),
                    "targets": tuple(config.targets),
                    "target_mean": y_mean.tolist(),
                    "target_std": y_std.tolist(),
                    "dataset_summary": asdict(dataset_summary),
                    "split": {
                        "train_subjects": split.train_subjects,
                        "val_subjects": split.val_subjects,
                        "test_subjects": split.test_subjects,
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= config.patience:
                print(
                    f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_macro_mae={best_val_metric:.4f}",
                    flush=True,
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_pred_scaled = _predict_averaged(
        model,
        val_loader,
        device=device,
        output_activation=config.output_activation,
        target_mean=y_mean,
        target_std=y_std,
        targets=config.targets,
    )
    test_pred_scaled = _predict_averaged(
        model,
        test_loader,
        device=device,
        output_activation=config.output_activation,
        target_mean=y_mean,
        target_std=y_std,
        targets=config.targets,
    )
    val_pred = val_pred_scaled * y_std + y_mean
    test_pred = test_pred_scaled * y_std + y_mean
    selection_val_metrics = regression_metrics(y[split.val], val_pred, config.targets)
    selection_test_metrics = regression_metrics(y[split.test], test_pred, config.targets)
    refit_seen_val_metrics: dict[str, Any] | None = None
    refit_epochs = 0
    if config.refit_train_val:
        refit_epochs = int(best_epoch)
        refit_idx = np.concatenate([split.train, split.val])
        y_scaled_refit, y_mean_refit, y_std_refit = scale_rating_targets(
            y,
            refit_idx,
            targets=config.targets,
            transform=config.target_transform,
        )
        refit_train_ds = PpgWindowDataset(
            examples,
            refit_idx,
            y_scaled_refit,
            target_length=config.target_length,
            min_window_seconds=config.min_window_seconds,
            max_window_seconds=config.max_window_seconds,
            eval_window_seconds=config.eval_window_seconds,
            train_window_anchor=config.train_window_anchor,
            input_normalization=config.input_normalization,
            input_clip_value=config.input_clip_value,
            input_noise_std=config.input_noise_std,
            time_mask_fraction=config.time_mask_fraction,
            windows_per_trial=config.train_windows_per_trial,
            train=True,
            seed=config.seed,
            emotion_label_column=emotion_label_column,
        )
        refit_val_ds = PpgWindowDataset(
            examples,
            split.val,
            y_scaled_refit,
            target_length=config.target_length,
            min_window_seconds=config.min_window_seconds,
            max_window_seconds=config.max_window_seconds,
            eval_window_seconds=config.eval_window_seconds,
            input_normalization=config.input_normalization,
            input_clip_value=config.input_clip_value,
            windows_per_trial=config.eval_windows_per_trial,
            train=False,
            seed=config.seed,
            emotion_label_column=emotion_label_column,
        )
        refit_test_ds = PpgWindowDataset(
            examples,
            split.test,
            y_scaled_refit,
            target_length=config.target_length,
            min_window_seconds=config.min_window_seconds,
            max_window_seconds=config.max_window_seconds,
            eval_window_seconds=config.eval_window_seconds,
            input_normalization=config.input_normalization,
            input_clip_value=config.input_clip_value,
            windows_per_trial=config.eval_windows_per_trial,
            train=False,
            seed=config.seed,
            emotion_label_column=emotion_label_column,
        )
        refit_loader = DataLoader(
            refit_train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
        )
        refit_val_loader = DataLoader(
            refit_val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
        )
        refit_test_loader = DataLoader(
            refit_test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
        )
        model = create_rating_model(
            config.model_arch,
            n_outputs=len(config.targets),
            shared_dropout=config.shared_dropout,
            head_dropout=config.head_dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, refit_epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device == "cuda")
        ema_state = clone_state_dict(model.state_dict()) if config.ema_decay > 0 else None
        for _ in range(refit_epochs):
            _run_epoch(
                model,
                refit_loader,
                device=device,
                loss_name=config.loss_name,
                output_activation=config.output_activation,
                ordinal_aux_weight=config.ordinal_aux_weight,
                emotion_aux_weight=config.emotion_aux_weight,
                mixup_alpha=config.mixup_alpha,
                target_mean=y_mean_refit,
                target_std=y_std_refit,
                targets=config.targets,
                target_weights=config.target_weights,
                variance_loss_weight=config.variance_loss_weight,
                variance_target_weights=config.variance_target_weights,
                correlation_loss_weight=config.correlation_loss_weight,
                correlation_target_weights=config.correlation_target_weights,
                optimizer=optimizer,
                scaler=scaler,
                ema_state=ema_state,
                ema_decay=config.ema_decay,
            )
            scheduler.step()
        if ema_state is not None:
            model.load_state_dict(ema_state)
        val_pred_scaled = _predict_averaged(
            model,
            refit_val_loader,
            device=device,
            output_activation=config.output_activation,
            target_mean=y_mean_refit,
            target_std=y_std_refit,
            targets=config.targets,
        )
        test_pred_scaled = _predict_averaged(
            model,
            refit_test_loader,
            device=device,
            output_activation=config.output_activation,
            target_mean=y_mean_refit,
            target_std=y_std_refit,
            targets=config.targets,
        )
        val_pred = val_pred_scaled * y_std_refit + y_mean_refit
        test_pred = test_pred_scaled * y_std_refit + y_mean_refit
        refit_seen_val_metrics = regression_metrics(y[split.val], val_pred, config.targets)
        y_mean = y_mean_refit
        y_std = y_std_refit
        torch.save(
            {
                "model_state": clone_state_dict(model.state_dict(), device="cpu"),
                "config": asdict(config),
                "targets": tuple(config.targets),
                "target_mean": y_mean.tolist(),
                "target_std": y_std.tolist(),
                "dataset_summary": asdict(dataset_summary),
                "split": {
                    "train_subjects": split.train_subjects,
                    "val_subjects": split.val_subjects,
                    "test_subjects": split.test_subjects,
                    "refit_train_val_subjects": (*split.train_subjects, *split.val_subjects),
                },
            },
            checkpoint_path,
        )
    val_true = y[split.val]
    test_true = y[split.test]
    train_true = y[split.train]

    result = {
        "model": "ppg_1d_cnn",
        "model_arch": config.model_arch,
        "task": "rating_regression",
        "targets": list(config.targets),
        "split": "participant_grouped_train_val_test",
        "config": asdict(config),
        "dataset": asdict(dataset_summary),
        "rows": int(y.shape[0]),
        "target_length": int(config.target_length),
        "input_channels": 2,
        "window_seconds": {
            "min": float(config.min_window_seconds),
            "max": float(config.max_window_seconds),
            "eval": float(config.eval_window_seconds),
        },
        "train_windows_per_trial": int(config.train_windows_per_trial),
        "eval_windows_per_trial": int(config.eval_windows_per_trial),
        "split_counts": {
            "train_rows": int(split.train.size),
            "val_rows": int(split.val.size),
            "test_rows": int(split.test.size),
            "train_subjects": len(split.train_subjects),
            "val_subjects": len(split.val_subjects),
            "test_subjects": len(split.test_subjects),
        },
        "device": device_info,
        "best_epoch": int(best_epoch),
        "refit_train_val": bool(config.refit_train_val),
        "refit_epochs": int(refit_epochs),
        "best_val_loss_scaled": float(best_val_loss),
        "best_val_macro_mae": float(best_val_metric),
        "checkpoint_selection": "lowest validation raw macro MAE"
        + (" using EMA weights" if config.ema_decay > 0 else "")
        + ("; final checkpoint refit on train+validation subjects for best_epoch epochs" if config.refit_train_val else ""),
        "elapsed_seconds": float(time.time() - start),
        "val_metrics": selection_val_metrics,
        "selection_test_metrics_before_refit": selection_test_metrics,
        "refit_seen_val_metrics": refit_seen_val_metrics,
        "test_metrics": regression_metrics(test_true, test_pred, config.targets),
        "mean_baseline_val_metrics": mean_baseline_metrics(train_true, val_true, config.targets),
        "mean_baseline_test_metrics": mean_baseline_metrics(train_true, test_true, config.targets),
        "target_mean_train": {target: float(y_mean[i]) for i, target in enumerate(config.targets)},
        "target_std_train": {target: float(y_std[i]) for i, target in enumerate(config.targets)},
        "checkpoint_path": str(checkpoint_path),
        "git_commit": git_commit(),
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    result_path = output_dir / f"ppg_{config.model_arch}_ratings_result_{stamp}.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    history_path = output_dir / f"ppg_{config.model_arch}_ratings_history_{stamp}.jsonl"
    with history_path.open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["history_path"] = str(history_path)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved result {result_path}", flush=True)
    return result


class RatingOutputWrapper(nn.Module):
    """Wrap the trained model so ONNX returns ratings on the original scale."""

    def __init__(
        self,
        model: nn.Module,
        target_mean: Sequence[float],
        target_std: Sequence[float],
        targets: Sequence[str],
        *,
        output_activation: str = "linear",
        output_scale: Sequence[float] | None = None,
        output_bias: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_activation = output_activation
        self.targets = tuple(targets)
        self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("target_std", torch.tensor(target_std, dtype=torch.float32).view(1, -1))
        if output_scale is None:
            output_scale = [1.0] * len(target_mean)
        if output_bias is None:
            output_bias = [0.0] * len(target_mean)
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float32).view(1, -1))
        self.register_buffer("output_bias", torch.tensor(output_bias, dtype=torch.float32).view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_scaled = apply_output_activation_torch(
            self.model(x),
            output_activation=self.output_activation,
            target_mean=self.target_mean,
            target_std=self.target_std,
            targets=self.targets,
        )
        raw = pred_scaled * self.target_std + self.target_mean
        return raw * self.output_scale + self.output_bias


def export_rating_cnn_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    *,
    opset: int = 17,
    output_scale: Sequence[float] | None = None,
    output_bias: Sequence[float] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a trained rating CNN checkpoint to an Android-friendly ONNX file."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    targets = tuple(checkpoint["targets"])
    config = checkpoint["config"]
    target_length = int(config["target_length"])
    model_arch = config.get("model_arch", "cnn")
    model = create_rating_model(
        model_arch,
        n_outputs=len(targets),
        shared_dropout=float(config.get("shared_dropout", 0.25)),
        head_dropout=float(config.get("head_dropout", 0.20)),
    )
    model.load_state_dict(checkpoint["model_state"])
    wrapped = RatingOutputWrapper(
        model,
        checkpoint["target_mean"],
        checkpoint["target_std"],
        targets,
        output_activation=config.get("output_activation", "linear"),
        output_scale=output_scale,
        output_bias=output_bias,
    )
    wrapped.eval()
    out = Path(output_path) if output_path is not None else checkpoint_path.with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 2, target_length, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        dummy,
        out,
        input_names=["ppg_window"],
        output_names=["ratings"],
        dynamic_axes={"ppg_window": {0: "batch"}, "ratings": {0: "batch"}},
        opset_version=opset,
    )
    sidecar = {
        "onnx_path": str(out),
        "checkpoint_path": str(checkpoint_path),
        "model_arch": model_arch,
        "targets": list(targets),
        "input_name": "ppg_window",
        "input_shape": [1, 2, target_length],
        "input_dtype": "float32",
        "channel_0": "PPG window resampled to target_length and normalized according to input_normalization",
        "channel_1": "duration ratio filled with window_seconds / max_window_seconds",
        "input_clip_value": config.get("input_clip_value"),
        "input_normalization": config.get("input_normalization", "robust"),
        "output_name": "ratings",
        "output_order": list(targets),
        "output_scale": "original rating units; Android should clamp display to valid questionnaire ranges",
        "output_activation": config.get("output_activation", "linear"),
        "calibration_output_scale": [float(x) for x in output_scale] if output_scale is not None else [1.0] * len(targets),
        "output_bias": [float(x) for x in output_bias] if output_bias is not None else [0.0] * len(targets),
        "min_window_seconds": float(config["min_window_seconds"]),
        "max_window_seconds": float(config["max_window_seconds"]),
        "target_length": target_length,
        "opset": int(opset),
    }
    if calibration is not None:
        sidecar["calibration"] = calibration
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sidecar["metadata_path"] = str(metadata_path)
    return sidecar


class RatingEnsembleOutputWrapper(nn.Module):
    """Average multiple trained rating models on the original rating scale."""

    def __init__(self, members: Sequence[RatingOutputWrapper]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([member(x) for member in self.members], dim=0)
        return stacked.mean(dim=0)


def load_rating_output_wrapper(checkpoint_path: str | Path) -> tuple[RatingOutputWrapper, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    targets = tuple(checkpoint["targets"])
    config = checkpoint["config"]
    model_arch = config.get("model_arch", "cnn")
    model = create_rating_model(
        model_arch,
        n_outputs=len(targets),
        shared_dropout=float(config.get("shared_dropout", 0.25)),
        head_dropout=float(config.get("head_dropout", 0.20)),
    )
    model.load_state_dict(checkpoint["model_state"])
    wrapper = RatingOutputWrapper(
        model,
        checkpoint["target_mean"],
        checkpoint["target_std"],
        targets,
        output_activation=config.get("output_activation", "linear"),
    )
    wrapper.eval()
    return wrapper, {
        "checkpoint_path": str(checkpoint_path),
        "targets": list(targets),
        "config": config,
        "model_arch": model_arch,
    }


def export_rating_ensemble_onnx(
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    opset: int = 18,
) -> dict[str, Any]:
    """Export an averaged ensemble of compatible rating checkpoints."""

    members: list[RatingOutputWrapper] = []
    metadata: list[dict[str, Any]] = []
    targets: list[str] | None = None
    target_length: int | None = None
    for checkpoint_path in checkpoint_paths:
        wrapper, info = load_rating_output_wrapper(checkpoint_path)
        cfg = info["config"]
        if targets is None:
            targets = list(info["targets"])
            target_length = int(cfg["target_length"])
        elif targets != list(info["targets"]) or target_length != int(cfg["target_length"]):
            raise ValueError("All ensemble checkpoints must share targets and target_length")
        members.append(wrapper)
        metadata.append(info)
    if not members or targets is None or target_length is None:
        raise ValueError("At least one checkpoint is required")

    ensemble = RatingEnsembleOutputWrapper(members).eval()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 2, target_length, dtype=torch.float32)
    torch.onnx.export(
        ensemble,
        dummy,
        out,
        input_names=["ppg_window"],
        output_names=["ratings"],
        dynamic_axes={"ppg_window": {0: "batch"}, "ratings": {0: "batch"}},
        opset_version=opset,
    )
    sidecar = {
        "onnx_path": str(out),
        "checkpoints": [item["checkpoint_path"] for item in metadata],
        "model_arches": [item["model_arch"] for item in metadata],
        "targets": targets,
        "input_name": "ppg_window",
        "input_shape": [1, 2, target_length],
        "input_dtype": "float32",
        "output_name": "ratings",
        "output_order": targets,
        "output_scale": "original rating units",
        "target_length": target_length,
        "opset": int(opset),
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sidecar["metadata_path"] = str(metadata_path)
    return sidecar


def clone_state_dict(state_dict: dict[str, torch.Tensor], *, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone().to(device=device) if device is not None else value.detach().clone()
        for key, value in state_dict.items()
    }


def update_ema_state(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if not torch.is_floating_point(value):
                ema_state[key] = value.detach().clone()
                continue
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _run_epoch(
    model: "Any",
    loader: "Any",
    *,
    device: str,
    loss_name: str,
    output_activation: str,
    ordinal_aux_weight: float,
    emotion_aux_weight: float,
    mixup_alpha: float,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    targets: Sequence[str],
    target_weights: Sequence[float] | None,
    variance_loss_weight: float,
    variance_target_weights: Sequence[float] | None,
    correlation_loss_weight: float,
    correlation_target_weights: Sequence[float] | None,
    optimizer: "Any | None" = None,
    scaler: "Any | None" = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
) -> float:
    import torch

    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device).view(1, -1)
    std = torch.as_tensor(target_std, dtype=torch.float32, device=device).view(1, -1)
    ranges = torch.as_tensor([RATING_TARGET_RANGES[target] for target in targets], dtype=torch.float32, device=device).view(1, -1)
    weights = resolve_target_weights(targets, target_weights, device=device)
    variance_weights = resolve_target_weights(targets, variance_target_weights, device=device)
    correlation_weights = resolve_target_weights(targets, correlation_target_weights, device=device)
    for batch_items in loader:
        xb = batch_items[0]
        yb = batch_items[1]
        emotion_y = batch_items[2] if len(batch_items) > 2 else None
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if emotion_y is not None:
            emotion_y = emotion_y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            if train and mixup_alpha > 0 and xb.shape[0] > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                order = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1.0 - lam) * xb[order]
                yb = lam * yb + (1.0 - lam) * yb[order]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None and scaler.is_enabled()):
                pred = apply_output_activation_torch(
                    model(xb),
                    output_activation=output_activation,
                    target_mean=mean,
                    target_std=std,
                    targets=targets,
                )
                loss = rating_loss(
                    pred,
                    yb,
                    loss_name=loss_name,
                    target_mean=mean,
                    target_std=std,
                    target_ranges=ranges,
                    target_weights=weights,
                    variance_loss_weight=variance_loss_weight,
                    variance_target_weights=variance_weights,
                    correlation_loss_weight=correlation_loss_weight,
                    correlation_target_weights=correlation_weights,
                )
                if ordinal_aux_weight > 0 and hasattr(model, "ordinal_logits"):
                    loss = loss + float(ordinal_aux_weight) * ordinal_auxiliary_loss(
                        model.ordinal_logits(xb),
                        yb,
                        target_mean=mean,
                        target_std=std,
                        targets=targets,
                    )
                if emotion_aux_weight > 0 and emotion_y is not None and hasattr(model, "emotion_logits"):
                    loss = loss + float(emotion_aux_weight) * nn.functional.cross_entropy(
                        model.emotion_logits(xb),
                        emotion_y,
                    )
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                if ema_state is not None and ema_decay > 0:
                    update_ema_state(ema_state, model, ema_decay)
        batch = xb.shape[0]
        total_loss += float(loss.detach().cpu()) * batch
        total += batch
    return total_loss / max(1, total)


def rating_loss(
    pred_scaled: torch.Tensor,
    y_scaled: torch.Tensor,
    *,
    loss_name: str,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_ranges: torch.Tensor,
    target_weights: torch.Tensor,
    variance_loss_weight: float,
    variance_target_weights: torch.Tensor,
    correlation_loss_weight: float,
    correlation_target_weights: torch.Tensor,
) -> torch.Tensor:
    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return torch.mean(values * target_weights)

    if loss_name == "smooth_l1_scaled":
        return weighted_mean(nn.functional.smooth_l1_loss(pred_scaled, y_scaled, beta=0.5, reduction="none"))
    if loss_name == "mae_scaled":
        return weighted_mean(torch.abs(pred_scaled - y_scaled))
    if loss_name == "mae_raw":
        pred_raw = pred_scaled * target_std + target_mean
        y_raw = y_scaled * target_std + target_mean
        return weighted_mean(torch.abs(pred_raw - y_raw))
    if loss_name == "range_mae_raw":
        pred_raw = pred_scaled * target_std + target_mean
        y_raw = y_scaled * target_std + target_mean
        return weighted_mean(torch.abs(pred_raw - y_raw) / target_ranges)
    if loss_name in {"range_mae_var_raw", "range_mae_under_var_raw", "range_epsilon_mae_var_raw"}:
        pred_raw = pred_scaled * target_std + target_mean
        y_raw = y_scaled * target_std + target_mean
        abs_error = torch.abs(pred_raw - y_raw)
        if loss_name == "range_epsilon_mae_var_raw":
            tolerance = 0.05 * target_ranges
            base = weighted_mean(torch.clamp(abs_error - tolerance, min=0.0) / target_ranges)
        else:
            base = weighted_mean(abs_error / target_ranges)
        if pred_raw.shape[0] < 2:
            return base
        pred_std = torch.std(pred_raw / target_ranges, dim=0, unbiased=False)
        y_std = torch.std(y_raw / target_ranges, dim=0, unbiased=False).clamp_min(1e-4)
        if loss_name == "range_mae_under_var_raw":
            variance_gap = torch.clamp(y_std - pred_std, min=0.0)
        else:
            variance_gap = torch.abs(pred_std - y_std)
        variance_penalty = torch.mean((variance_gap / y_std) * variance_target_weights.view(-1))
        loss = base + float(variance_loss_weight) * variance_penalty
        if correlation_loss_weight > 0:
            pred_norm = pred_raw / target_ranges
            y_norm = y_raw / target_ranges
            pred_centered = pred_norm - pred_norm.mean(dim=0, keepdim=True)
            y_centered = y_norm - y_norm.mean(dim=0, keepdim=True)
            numerator = torch.mean(pred_centered * y_centered, dim=0)
            denominator = torch.sqrt(torch.mean(pred_centered.square(), dim=0) * torch.mean(y_centered.square(), dim=0) + 1e-8)
            corr = numerator / denominator
            corr_penalty = torch.mean((1.0 - corr) * correlation_target_weights.view(-1))
            loss = loss + float(correlation_loss_weight) * corr_penalty
        return loss
    if loss_name == "range_epsilon_mae_raw":
        pred_raw = pred_scaled * target_std + target_mean
        y_raw = y_scaled * target_std + target_mean
        tolerance = 0.05 * target_ranges
        return weighted_mean(torch.clamp(torch.abs(pred_raw - y_raw) - tolerance, min=0.0) / target_ranges)
    if loss_name == "smooth_l1_raw":
        pred_raw = pred_scaled * target_std + target_mean
        y_raw = y_scaled * target_std + target_mean
        return weighted_mean(nn.functional.smooth_l1_loss(pred_raw, y_raw, beta=0.5, reduction="none"))
    if loss_name == "range_smooth_l1_raw":
        pred_raw = pred_scaled * target_std + target_mean
        y_raw = y_scaled * target_std + target_mean
        return weighted_mean(nn.functional.smooth_l1_loss(pred_raw / target_ranges, y_raw / target_ranges, beta=0.125, reduction="none"))
    raise ValueError(f"Unsupported loss_name={loss_name!r}")


def ordinal_auxiliary_loss(
    logits_by_target: Sequence[torch.Tensor],
    y_scaled: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    targets: Sequence[str],
) -> torch.Tensor:
    y_raw = y_scaled * target_std + target_mean
    losses: list[torch.Tensor] = []
    for target_idx, target in enumerate(targets):
        target_min = float(RATING_TARGET_MINS[target])
        target_range = float(RATING_TARGET_RANGES[target])
        normalized = (y_raw[:, target_idx] - target_min) / max(target_range, 1e-6)
        labels = torch.clamp(torch.floor(normalized * 3.0), min=0, max=2).long()
        losses.append(nn.functional.cross_entropy(logits_by_target[target_idx], labels))
    return torch.stack(losses).mean()


def resolve_target_weights(targets: Sequence[str], target_weights: Sequence[float] | None, *, device: str) -> torch.Tensor:
    if target_weights is None:
        values = [1.0] * len(targets)
    else:
        values = [float(x) for x in target_weights]
        if len(values) != len(targets):
            raise ValueError(f"target_weights length {len(values)} does not match targets length {len(targets)}")
    arr = torch.as_tensor(values, dtype=torch.float32, device=device).view(1, -1)
    mean = torch.mean(arr)
    return arr / mean.clamp_min(1e-8)


def _predict(model: "Any", loader: "Any", *, device: str) -> np.ndarray:
    import torch

    model.eval()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_items in loader:
            xb = batch_items[0]
            xb = xb.to(device, non_blocking=True)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def _predict_averaged(
    model: "Any",
    loader: "Any",
    *,
    device: str,
    output_activation: str = "linear",
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    targets: Sequence[str] | None = None,
) -> np.ndarray:
    pred = _predict(model, loader, device=device)
    if output_activation != "linear":
        if target_mean is None or target_std is None or targets is None:
            raise ValueError("target_mean, target_std, and targets are required for bounded output prediction")
        pred = apply_output_activation_numpy(
            pred,
            output_activation=output_activation,
            target_mean=target_mean,
            target_std=target_std,
            targets=targets,
        )
    dataset = getattr(loader, "dataset", None)
    windows_per_trial = int(getattr(dataset, "windows_per_trial", 1))
    train = bool(getattr(dataset, "train", False))
    if train or windows_per_trial <= 1:
        return pred
    base_count = int(getattr(dataset, "indices").size)
    return pred.reshape(windows_per_trial, base_count, pred.shape[-1]).mean(axis=0)


def conv_block(in_channels: int, out_channels: int, *, kernel_size: int, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.SiLU(inplace=True),
    )


class DilatedResidualBlock(nn.Module):
    """Residual temporal block for ID-CNN-style long-range PPG context."""

    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 3
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualConvBlock(nn.Module):
    """Downsampling residual 1D convolution block for compact PPG encoders."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


def create_rating_model(
    model_arch: str,
    *,
    n_outputs: int,
    shared_dropout: float = 0.25,
    head_dropout: float = 0.20,
) -> nn.Module:
    if model_arch == "cnn":
        return PpgRatingCnn(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_gru":
        return PpgRatingCnnGru(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_lstm":
        return PpgRatingCnnLstm(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_multihead":
        return PpgRatingCnnMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_attention_multihead":
        return PpgRatingCnnAttentionMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_derivative_multihead":
        return PpgRatingCnnDerivativeMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_emotion_multihead":
        return PpgRatingCnnEmotionMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_ordinal_multihead":
        return PpgRatingCnnOrdinalMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "cnn_statfusion_multihead":
        return PpgRatingCnnStatFusionMultiHead(
            n_outputs=n_outputs,
            shared_dropout=shared_dropout,
            head_dropout=head_dropout,
        )
    if model_arch == "cnn_transformer_multihead":
        return PpgRatingCnnTransformerMultiHead(
            n_outputs=n_outputs,
            shared_dropout=shared_dropout,
            head_dropout=head_dropout,
        )
    if model_arch == "idcnn_multihead":
        return PpgRatingIdCnnMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "tcn_multihead":
        return PpgRatingTcnMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    if model_arch == "resnet_multihead":
        return PpgRatingResNetMultiHead(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
    raise ValueError(
        f"Unsupported model_arch={model_arch!r}; use 'cnn', 'cnn_gru', 'cnn_lstm', "
        "'cnn_multihead', 'cnn_attention_multihead', 'cnn_derivative_multihead', "
        "'cnn_emotion_multihead', 'cnn_ordinal_multihead', 'cnn_statfusion_multihead', "
        "'cnn_transformer_multihead', 'idcnn_multihead', 'tcn_multihead', or 'resnet_multihead'."
    )


class PpgRatingCnn(nn.Module):
    """Compact multi-output 1D-CNN for raw PPG rating regression."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
            conv_block(128, 160, kernel_size=3, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(320),
            nn.Dropout(shared_dropout),
            nn.Linear(320, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1, keepdim=True), z.amax(dim=-1, keepdim=True)), dim=1)
        return self.head(pooled)


class PpgRatingCnnMultiHead(nn.Module):
    """Shared 1D-CNN encoder with independent per-target regression heads."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
            conv_block(128, 160, kernel_size=3, stride=2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(320),
            nn.Dropout(shared_dropout),
            nn.Linear(320, 160),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1, keepdim=True), z.amax(dim=-1, keepdim=True)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingCnnAttentionMultiHead(nn.Module):
    """CNN multihead regressor with learned temporal attention pooling."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
            conv_block(128, 160, kernel_size=3, stride=2),
        )
        self.attention = nn.Conv1d(160, 1, kernel_size=1)
        self.shared = nn.Sequential(
            nn.LayerNorm(480),
            nn.Dropout(shared_dropout),
            nn.Linear(480, 192),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(192, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        weights = torch.softmax(self.attention(z), dim=-1)
        attended = torch.sum(z * weights, dim=-1)
        pooled = torch.cat((attended, z.mean(dim=-1), z.amax(dim=-1)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


def add_ppg_derivative_channel(x: torch.Tensor) -> torch.Tensor:
    """Append a first-difference PPG channel while preserving the 2-channel input contract."""

    ppg = x[:, 0:1, :]
    derivative = torch.diff(ppg, dim=-1, prepend=ppg[:, :, :1])
    duration = x[:, 1:2, :]
    return torch.cat((ppg, derivative, duration), dim=1)


class PpgRatingCnnDerivativeMultiHead(nn.Module):
    """CNN multihead regressor that derives a local-slope channel inside the model."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
            conv_block(128, 160, kernel_size=3, stride=2),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(320),
            nn.Dropout(shared_dropout),
            nn.Linear(320, 160),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(add_ppg_derivative_channel(x))
        pooled = torch.cat((z.mean(dim=-1, keepdim=True), z.amax(dim=-1, keepdim=True)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingCnnOrdinalMultiHead(PpgRatingCnnMultiHead):
    """CNN multihead regressor with auxiliary low/mid/high ordinal heads for training."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
        self.ordinal_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 3),
                )
                for _ in range(n_outputs)
            ]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1, keepdim=True), z.amax(dim=-1, keepdim=True)), dim=1)
        return self.shared(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.encode(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)

    def ordinal_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        shared = self.encode(x)
        return [head(shared) for head in self.ordinal_heads]


class PpgRatingCnnEmotionMultiHead(PpgRatingCnnMultiHead):
    """CNN multihead regressor with an auxiliary five-class emotion head."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__(n_outputs=n_outputs, shared_dropout=shared_dropout, head_dropout=head_dropout)
        self.emotion_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(160, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 5),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1, keepdim=True), z.amax(dim=-1, keepdim=True)), dim=1)
        return self.shared(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.encode(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)

    def emotion_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.emotion_head(self.encode(x))


def ppg_summary_stats(x: torch.Tensor) -> torch.Tensor:
    """ONNX-friendly summary features from the normalized PPG input."""

    ppg = x[:, 0, :]
    duration = x[:, 1, :].mean(dim=-1, keepdim=True)
    mean = ppg.mean(dim=-1, keepdim=True)
    centered = ppg - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    std = torch.sqrt(variance + 1e-6)
    mean_abs = ppg.abs().mean(dim=-1, keepdim=True)
    rms = torch.sqrt((ppg * ppg).mean(dim=-1, keepdim=True) + 1e-6)
    min_value = ppg.amin(dim=-1, keepdim=True)
    max_value = ppg.amax(dim=-1, keepdim=True)
    value_range = max_value - min_value
    return torch.cat((duration, mean, std, mean_abs, rms, min_value, max_value, value_range), dim=1)


class PpgRatingCnnStatFusionMultiHead(nn.Module):
    """CNN multihead model with ONNX-friendly per-window PPG statistics."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
            conv_block(128, 160, kernel_size=3, stride=2),
        )
        self.stat_norm = nn.LayerNorm(8)
        self.shared = nn.Sequential(
            nn.LayerNorm(328),
            nn.Dropout(shared_dropout),
            nn.Linear(328, 176),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(176, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1, keepdim=True), z.amax(dim=-1, keepdim=True)), dim=1)
        stats = self.stat_norm(ppg_summary_stats(x))
        shared = self.shared(torch.cat((pooled.flatten(start_dim=1), stats), dim=1))
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingResNetMultiHead(nn.Module):
    """Residual temporal CNN with independent per-target regression heads."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualConvBlock(2, 32, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            ResidualConvBlock(32, 64, kernel_size=11, stride=2),
            ResidualConvBlock(64, 96, kernel_size=9, stride=2),
            ResidualConvBlock(96, 128, kernel_size=7, stride=2),
            ResidualConvBlock(128, 160, kernel_size=5, stride=2),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(320),
            nn.Dropout(shared_dropout),
            nn.Linear(320, 160),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1), z.amax(dim=-1)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingCnnTransformerMultiHead(nn.Module):
    """CNN downsampler plus Transformer encoder with independent regression heads."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 32, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(32, 64, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(64, 128, kernel_size=7, stride=2),
            conv_block(128, 128, kernel_size=5, stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=shared_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.shared = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(shared_dropout),
            nn.Linear(256, 160),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).transpose(1, 2)
        seq = self.encoder(z)
        pooled = torch.cat((seq.mean(dim=1), seq.amax(dim=1)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingIdCnnMultiHead(nn.Module):
    """ID-CNN-style dilated encoder with independent per-target regression heads."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(2, 32, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(32, 64, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(64, 128, kernel_size=7, stride=2),
        )
        self.dilated = nn.Sequential(
            DilatedResidualBlock(128, dilation=1, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=2, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=4, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=8, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=1, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=2, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=4, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=8, dropout=shared_dropout * 0.5),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(shared_dropout),
            nn.Linear(256, 160),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dilated(self.stem(x))
        pooled = torch.cat((z.mean(dim=-1), z.amax(dim=-1)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingTcnMultiHead(nn.Module):
    """TCN-lite encoder that preserves more temporal resolution before dilated pooling."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(2, 32, kernel_size=15, stride=2),
            conv_block(32, 64, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(64, 128, kernel_size=7, stride=2),
        )
        self.temporal = nn.Sequential(
            DilatedResidualBlock(128, dilation=1, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=2, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=4, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=8, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=16, dropout=shared_dropout * 0.5),
            DilatedResidualBlock(128, dilation=1, dropout=shared_dropout * 0.5),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(shared_dropout),
            nn.Linear(256, 160),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(160, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 1),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.temporal(self.stem(x))
        pooled = torch.cat((z.mean(dim=-1), z.amax(dim=-1)), dim=1)
        shared = self.shared(pooled)
        return torch.cat([head(shared) for head in self.heads], dim=1)


class PpgRatingCnnGru(nn.Module):
    """1D-CNN encoder plus GRU temporal head for streaming-window rating regression."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(shared_dropout),
            nn.Linear(384, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).transpose(1, 2)
        seq, _ = self.gru(z)
        pooled = torch.cat((seq.mean(dim=1), seq.amax(dim=1)), dim=1)
        return self.head(pooled)


class PpgRatingCnnLstm(nn.Module):
    """1D-CNN encoder plus LSTM temporal head for streaming-window rating regression."""

    def __init__(self, n_outputs: int, *, shared_dropout: float = 0.25, head_dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(2, 24, kernel_size=15, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(24, 48, kernel_size=11, stride=2),
            nn.MaxPool1d(kernel_size=2),
            conv_block(48, 96, kernel_size=7, stride=2),
            conv_block(96, 128, kernel_size=5, stride=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(shared_dropout),
            nn.Linear(384, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).transpose(1, 2)
        seq, _ = self.lstm(z)
        pooled = torch.cat((seq.mean(dim=1), seq.amax(dim=1)), dim=1)
        return self.head(pooled)
