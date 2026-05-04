#!/usr/bin/env python
"""Build trial-wise LLaMAC feature tables from extracted participant folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llamac_research.features import build_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/extracted", help="Extracted LLaMAC participant directory root.")
    parser.add_argument(
        "--mode",
        choices=["all", "ppg", "ppg_rich"],
        default="all",
        help="Feature mode: all biosignal modalities, base PPG, or rich PPG-only features.",
    )
    parser.add_argument("--output", default=None, help="Output CSV/parquet path.")
    parser.add_argument("--limit-subjects", type=int, default=None, help="Optional smoke-test participant limit.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel participant workers; 1 is deterministic and gentle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = f"data/processed/features_{args.mode}.parquet"
    _, summary = build_feature_table(
        args.data_root,
        mode=args.mode,
        limit_subjects=args.limit_subjects,
        workers=args.workers,
        output_path=output,
    )
    summary_path = Path(output).with_suffix(Path(output).suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
