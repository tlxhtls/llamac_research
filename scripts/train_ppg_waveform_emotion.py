#!/usr/bin/env python
"""Train raw-PPG waveform classifiers for five-class LLaMAC emotion labels."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from llamac_research import __version__
from llamac_research.labels import EMOTION_ID_TO_LABEL, EMOTION_IDS
from llamac_research.metrics import compute_classification_metrics, metrics_summary_line
from llamac_research.waveform_dnn import (
    RATING_TARGETS,
    WaveformExample,
    clone_state_dict,
    create_rating_model,
    crop_ppg_window,
    git_commit,
    load_ppg_rating_examples,
    make_subject_split,
    select_torch_device,
    set_reproducible_seed,
    window_to_model_input,
)


MODEL_ARCHES = (
    "cnn",
    "cnn_gru",
    "cnn_lstm",
    "cnn_multihead",
    "cnn_attention_multihead",
    "cnn_derivative_multihead",
    "cnn_statfusion_multihead",
    "cnn_transformer_multihead",
    "idcnn_multihead",
    "tcn_multihead",
    "resnet_multihead",
)


@dataclass(frozen=True)
class EmotionCnnConfig:
    data_root: str = "data/extracted"
    output_dir: str = "artifacts/results"
    label_column: Literal["ReportedType", "IntendedType"] = "ReportedType"
    model_arch: str = "cnn"
    target_length: int = 1920
    min_window_seconds: float = 30.0
    max_window_seconds: float = 30.0
    eval_window_seconds: float = 30.0
    train_window_anchor: Literal["random", "first", "center", "last", "even"] = "random"
    input_normalization: Literal["robust", "zscore", "none"] = "robust"
    input_clip_value: float | None = None
    train_windows_per_trial: int = 4
    eval_windows_per_trial: int = 1
    seed: int = 777
    split_seed: int | None = 42
    batch_size: int = 512
    epochs: int = 80
    patience: int = 14
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    shared_dropout: float = 0.25
    head_dropout: float = 0.20
    label_smoothing: float = 0.02
    class_weight: Literal["balanced", "sqrt_balanced", "none"] = "sqrt_balanced"
    official_valid_only: bool = False
    val_subject_fraction: float = 0.15
    test_subject_fraction: float = 0.15
    max_subjects: int | None = None
    num_workers: int = 4
    device: Literal["auto", "cuda", "cpu"] = "auto"
    amp: bool = True


class PpgEmotionWindowDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Streaming-style raw PPG windows paired with a five-class emotion label."""

    def __init__(
        self,
        examples: list[WaveformExample],
        indices: np.ndarray,
        labels: np.ndarray,
        *,
        label_column: str,
        target_length: int,
        min_window_seconds: float,
        max_window_seconds: float,
        eval_window_seconds: float,
        train_window_anchor: Literal["random", "first", "center", "last", "even"],
        input_normalization: Literal["robust", "zscore", "none"],
        input_clip_value: float | None,
        windows_per_trial: int,
        train: bool,
        seed: int,
    ) -> None:
        self.examples = examples
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels.astype(np.int64, copy=False)
        self.label_column = label_column
        self.target_length = int(target_length)
        self.min_window_seconds = float(min_window_seconds)
        self.max_window_seconds = float(max_window_seconds)
        self.eval_window_seconds = float(eval_window_seconds)
        self.train_window_anchor = train_window_anchor
        self.input_normalization = input_normalization
        self.input_clip_value = input_clip_value
        self.windows_per_trial = max(1, int(windows_per_trial))
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.indices.size * self.windows_per_trial)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            anchor = "center"
            rng = None
            start_fraction = 1.0 if self.windows_per_trial == 1 else repeat_idx / max(1, self.windows_per_trial - 1)
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
        return torch.from_numpy(x), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def _emotion_arrays(examples: list[WaveformExample], label_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.empty(len(examples), dtype=np.int64)
    subjects = np.empty(len(examples), dtype=object)
    trials = np.empty(len(examples), dtype=np.int64)
    for idx, example in enumerate(examples):
        labels[idx] = int(example.labels[label_column]) - 1
        subjects[idx] = example.subject_id
        trials[idx] = int(example.trial)
    invalid = (labels < 0) | (labels >= len(EMOTION_IDS))
    if np.any(invalid):
        bad = [f"{examples[idx].subject_id}:{examples[idx].trial}" for idx in np.flatnonzero(invalid)[:5]]
        raise ValueError(f"Invalid {label_column} values in examples: {bad}")
    return labels, subjects, trials


def _class_weight_tensor(labels: np.ndarray, mode: str, *, device: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels.astype(int), minlength=len(EMOTION_IDS)).astype(np.float64)
    counts[counts <= 0] = 1.0
    weights = labels.size / (len(EMOTION_IDS) * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    weights = weights / np.mean(weights)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(int), minlength=len(EMOTION_IDS))
    return {EMOTION_ID_TO_LABEL[idx + 1]: int(counts[idx]) for idx in range(len(EMOTION_IDS))}


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=scaler is not None and scaler.is_enabled(),
            ):
                logits = model(xb)
                loss = criterion(logits, yb)
            if train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * int(xb.shape[0])
        total += int(xb.shape[0])
    return total_loss / max(1, total)


def _predict_probs(model: nn.Module, loader: torch.utils.data.DataLoader, *, device: str) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            prob = torch.softmax(model(xb), dim=1).detach().cpu().numpy()
            probs.append(prob)
    pred = np.concatenate(probs, axis=0)
    dataset = getattr(loader, "dataset", None)
    windows_per_trial = int(getattr(dataset, "windows_per_trial", 1))
    if windows_per_trial <= 1:
        return pred
    base_count = int(getattr(dataset, "indices").size)
    return pred.reshape(windows_per_trial, base_count, pred.shape[-1]).mean(axis=0)


def _metrics_from_probs(labels_zero_based: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y_true = labels_zero_based.astype(int) + 1
    y_pred = np.argmax(probs, axis=1).astype(int) + 1
    return compute_classification_metrics(y_true, y_pred, probs, labels=EMOTION_IDS).to_dict()


def _prior_baseline(train_labels: np.ndarray, eval_labels: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(train_labels.astype(int), minlength=len(EMOTION_IDS)).astype(np.float64)
    prior = counts / counts.sum()
    probs = np.broadcast_to(prior.reshape(1, -1), (eval_labels.size, prior.size))
    return _metrics_from_probs(eval_labels, probs)


def train_emotion_cnn(config: EmotionCnnConfig) -> dict[str, Any]:
    if config.model_arch not in MODEL_ARCHES:
        raise ValueError(f"Unsupported model_arch={config.model_arch!r}; choices={MODEL_ARCHES}")
    set_reproducible_seed(config.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device, device_info = select_torch_device(config.device)
    examples, dataset_summary = load_ppg_rating_examples(
        config.data_root,
        targets=RATING_TARGETS,
        max_subjects=config.max_subjects,
        official_valid_only=config.official_valid_only,
    )
    labels, subjects, _trials = _emotion_arrays(examples, config.label_column)
    split = make_subject_split(
        subjects,
        seed=config.split_seed if config.split_seed is not None else config.seed,
        val_fraction=config.val_subject_fraction,
        test_fraction=config.test_subject_fraction,
    )
    train_ds = PpgEmotionWindowDataset(
        examples,
        split.train,
        labels,
        label_column=config.label_column,
        target_length=config.target_length,
        min_window_seconds=config.min_window_seconds,
        max_window_seconds=config.max_window_seconds,
        eval_window_seconds=config.eval_window_seconds,
        train_window_anchor=config.train_window_anchor,
        input_normalization=config.input_normalization,
        input_clip_value=config.input_clip_value,
        windows_per_trial=config.train_windows_per_trial,
        train=True,
        seed=config.seed,
    )
    val_ds = PpgEmotionWindowDataset(
        examples,
        split.val,
        labels,
        label_column=config.label_column,
        target_length=config.target_length,
        min_window_seconds=config.min_window_seconds,
        max_window_seconds=config.max_window_seconds,
        eval_window_seconds=config.eval_window_seconds,
        train_window_anchor=config.train_window_anchor,
        input_normalization=config.input_normalization,
        input_clip_value=config.input_clip_value,
        windows_per_trial=config.eval_windows_per_trial,
        train=False,
        seed=config.seed,
    )
    test_ds = PpgEmotionWindowDataset(
        examples,
        split.test,
        labels,
        label_column=config.label_column,
        target_length=config.target_length,
        min_window_seconds=config.min_window_seconds,
        max_window_seconds=config.max_window_seconds,
        eval_window_seconds=config.eval_window_seconds,
        train_window_anchor=config.train_window_anchor,
        input_normalization=config.input_normalization,
        input_clip_value=config.input_clip_value,
        windows_per_trial=config.eval_windows_per_trial,
        train=False,
        seed=config.seed,
    )
    pin_memory = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    model = create_rating_model(
        config.model_arch,
        n_outputs=len(EMOTION_IDS),
        shared_dropout=config.shared_dropout,
        head_dropout=config.head_dropout,
    ).to(device)
    weights = _class_weight_tensor(labels[split.train], config.class_weight, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(config.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device == "cuda")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label_slug = "reported" if config.label_column == "ReportedType" else "intended"
    checkpoint_path = output_dir / f"ppg_{config.model_arch}_{label_slug}_emotion_best_{stamp}.pt"
    result_path = output_dir / f"ppg_{config.model_arch}_{label_slug}_emotion_result_{stamp}.json"
    history_path = output_dir / f"ppg_{config.model_arch}_{label_slug}_emotion_history_{stamp}.jsonl"

    best_val_macro_f1 = -math.inf
    best_val_top1 = -math.inf
    best_epoch = 0
    best_val_loss = math.inf
    stale = 0
    history: list[dict[str, float | int]] = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device=device, criterion=criterion, optimizer=optimizer, scaler=scaler)
        val_loss = _run_epoch(model, val_loader, device=device, criterion=criterion)
        val_probs = _predict_probs(model, val_loader, device=device)
        val_metrics = _metrics_from_probs(labels[split.val], val_probs)
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_top1 = float(val_metrics["top1_accuracy"])
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_top1_accuracy": val_top1,
            "val_top2_accuracy": float(val_metrics["top2_accuracy"]),
            "val_top3_accuracy": float(val_metrics["top3_accuracy"]),
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            " ".join(
                [
                    f"epoch={epoch:03d}",
                    f"train_loss={train_loss:.4f}",
                    f"val_loss={val_loss:.4f}",
                    f"val_top1={val_top1:.4f}",
                    f"val_top3={val_metrics['top3_accuracy']:.4f}",
                    f"val_macro_f1={val_macro_f1:.4f}",
                    f"lr={row['lr']:.2e}",
                ]
            ),
            flush=True,
        )
        improved = (val_macro_f1 > best_val_macro_f1 + 1e-5) or (
            abs(val_macro_f1 - best_val_macro_f1) <= 1e-5 and val_top1 > best_val_top1 + 1e-5
        )
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_val_top1 = val_top1
            best_val_loss = float(val_loss)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": clone_state_dict(model.state_dict(), device="cpu"),
                    "config": asdict(config),
                    "label_column": config.label_column,
                    "class_labels": EMOTION_IDS,
                    "class_names": [EMOTION_ID_TO_LABEL[label] for label in EMOTION_IDS],
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
                    f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_macro_f1={best_val_macro_f1:.4f}",
                    flush=True,
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_probs = _predict_probs(model, val_loader, device=device)
    test_probs = _predict_probs(model, test_loader, device=device)
    val_metrics = _metrics_from_probs(labels[split.val], val_probs)
    test_metrics = _metrics_from_probs(labels[split.test], test_probs)
    train_labels = labels[split.train]
    result = {
        "model": "ppg_waveform_emotion_classifier",
        "model_arch": config.model_arch,
        "task": "emotion_classification",
        "target": config.label_column,
        "class_labels": EMOTION_IDS,
        "class_names": [EMOTION_ID_TO_LABEL[label] for label in EMOTION_IDS],
        "split": "participant_grouped_train_val_test",
        "config": asdict(config),
        "strict_conditions": {
            "ppg_only": True,
            "single_fixed_eval_window": config.eval_windows_per_trial == 1,
            "train_window_seconds": [float(config.min_window_seconds), float(config.max_window_seconds)],
            "eval_window_seconds": float(config.eval_window_seconds),
            "participant_grouped": True,
            "full_signal_or_trial_aggregation": False,
        },
        "dataset": asdict(dataset_summary),
        "rows": int(labels.size),
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
        "label_distribution": {
            "all": _distribution(labels),
            "train": _distribution(labels[split.train]),
            "val": _distribution(labels[split.val]),
            "test": _distribution(labels[split.test]),
        },
        "device": device_info,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "checkpoint_selection": "highest validation macro F1, top-1 tie-break",
        "elapsed_seconds": float(time.time() - start),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "prior_baseline_val_metrics": _prior_baseline(train_labels, labels[split.val]),
        "prior_baseline_test_metrics": _prior_baseline(train_labels, labels[split.test]),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "result_path": str(result_path),
        "git_commit": git_commit(),
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    with history_path.open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"test_metrics {metrics_summary_line(test_metrics)}", flush=True)
    print(f"saved result {result_path}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/extracted")
    parser.add_argument("--output-dir", default="artifacts/results")
    parser.add_argument("--label-column", choices=["ReportedType", "IntendedType"], default="ReportedType")
    parser.add_argument("--model-arch", choices=MODEL_ARCHES, default="cnn")
    parser.add_argument("--target-length", type=int, default=1920)
    parser.add_argument("--min-window-seconds", type=float, default=30.0)
    parser.add_argument("--max-window-seconds", type=float, default=30.0)
    parser.add_argument("--eval-window-seconds", type=float, default=30.0)
    parser.add_argument("--train-window-anchor", choices=["random", "first", "center", "last", "even"], default="random")
    parser.add_argument("--input-normalization", choices=["robust", "zscore", "none"], default="robust")
    parser.add_argument("--input-clip-value", type=float, default=None)
    parser.add_argument("--train-windows-per-trial", type=int, default=4)
    parser.add_argument("--eval-windows-per-trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--shared-dropout", type=float, default=0.25)
    parser.add_argument("--head-dropout", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--class-weight", choices=["balanced", "sqrt_balanced", "none"], default="sqrt_balanced")
    parser.add_argument("--official-valid-only", action="store_true")
    parser.add_argument("--val-subject-fraction", type=float, default=0.15)
    parser.add_argument("--test-subject-fraction", type=float, default=0.15)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmotionCnnConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        label_column=args.label_column,
        model_arch=args.model_arch,
        target_length=args.target_length,
        min_window_seconds=args.min_window_seconds,
        max_window_seconds=args.max_window_seconds,
        eval_window_seconds=args.eval_window_seconds,
        train_window_anchor=args.train_window_anchor,
        input_normalization=args.input_normalization,
        input_clip_value=args.input_clip_value,
        train_windows_per_trial=args.train_windows_per_trial,
        eval_windows_per_trial=args.eval_windows_per_trial,
        seed=args.seed,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        shared_dropout=args.shared_dropout,
        head_dropout=args.head_dropout,
        label_smoothing=args.label_smoothing,
        class_weight=args.class_weight,
        official_valid_only=args.official_valid_only,
        val_subject_fraction=args.val_subject_fraction,
        test_subject_fraction=args.test_subject_fraction,
        max_subjects=args.max_subjects,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
    )
    train_emotion_cnn(config)


if __name__ == "__main__":
    main()
