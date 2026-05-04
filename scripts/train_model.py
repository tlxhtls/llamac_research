#!/usr/bin/env python
"""Train/evaluate one LLaMAC emotion model with grouped or paper-style CV."""

from __future__ import annotations

import argparse

from llamac_research.modeling import ExperimentConfig, MODEL_NAMES, run_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Feature table CSV/parquet from scripts/build_features.py.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="lightgbm")
    parser.add_argument("--feature-set", choices=["all", "ppg"], default="all")
    parser.add_argument("--target", choices=["reported", "intended"], default="reported")
    parser.add_argument("--split", choices=["grouped", "stratified"], default="grouped")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--official-exclusions",
        action="store_true",
        help="Apply the official notebook's invalid subject/trial exclusions before training.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap.")
    parser.add_argument("--output-dir", default="artifacts/results")
    parser.add_argument("--output", default=None, help="Explicit result JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        feature_path=args.features,
        model_name=args.model,
        feature_set=args.feature_set,
        target=args.target,
        split_strategy=args.split,
        n_splits=args.folds,
        seed=args.seed,
        device=args.device,
        apply_official_exclusions=args.official_exclusions,
        max_rows=args.max_rows,
        output_dir=args.output_dir,
    )
    run_and_save(config, output_path=args.output)


if __name__ == "__main__":
    main()
