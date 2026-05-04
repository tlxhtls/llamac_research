#!/usr/bin/env python
"""Summarize LLaMAC result JSON files into a compact CSV/Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRIC_KEYS = [
    "top1_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "cohen_kappa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Result JSON files or directories containing JSON files.")
    parser.add_argument("--output", default="artifacts/results/summary.csv")
    return parser.parse_args()


def iter_json(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from sorted(p.rglob("*.json"))
        else:
            yield p


def main() -> None:
    args = parse_args()
    rows = []
    for path in iter_json(args.paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "metrics" not in payload or "config" not in payload:
            continue
        config = payload["config"]
        metrics = payload["metrics"]
        row = {
            "path": str(path),
            "model": config.get("model_name"),
            "feature_set": config.get("feature_set"),
            "target": config.get("target"),
            "split": config.get("split_strategy"),
            "official_exclusions": config.get("apply_official_exclusions"),
            "rows": payload.get("data", {}).get("rows"),
            "features": payload.get("data", {}).get("candidate_features"),
            "backend": payload.get("backend", {}).get("selected"),
        }
        row.update({key: metrics.get(key) for key in METRIC_KEYS})
        rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "feature_set",
        "target",
        "split",
        "official_exclusions",
        "rows",
        "features",
        "backend",
        *METRIC_KEYS,
        "path",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] rows={len(rows)} output={out}")


if __name__ == "__main__":
    main()
