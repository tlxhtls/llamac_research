#!/usr/bin/env python
"""Run Optuna studies for one or more LLaMAC model families."""

from __future__ import annotations

import argparse

from llamac_research.modeling import MODEL_NAMES
from llamac_research.tuning import TuningConfig, run_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature table CSV/parquet from scripts/build_features.py.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["lightgbm", "extra_trees", "hist_gradient_boosting"],
        help="Model families to tune.",
    )
    parser.add_argument("--feature-set", choices=["all", "ppg"], default="ppg")
    parser.add_argument("--target", choices=["reported", "intended"], default="reported")
    parser.add_argument("--split", choices=["grouped", "stratified"], default="grouped")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--official-exclusions", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--output-dir", default="artifacts/optuna")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL; defaults to ignored local sqlite files.")
    parser.add_argument("--study-prefix", default="llamac")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TuningConfig(
        feature_path=args.features,
        models=list(args.models),
        feature_set=args.feature_set,
        target=args.target,
        split_strategy=args.split,
        n_splits=args.folds,
        seed=args.seed,
        device=args.device,
        apply_official_exclusions=args.official_exclusions,
        n_trials=args.trials,
        timeout_seconds=args.timeout_seconds,
        metric=args.metric,
        output_dir=args.output_dir,
        storage=args.storage,
        study_prefix=args.study_prefix,
        max_rows=args.max_rows,
    )
    run_tuning(config)


if __name__ == "__main__":
    main()
