# Reproducible LLaMAC modeling workflow

This page records the public commands for reproducing the first modeling stage:
source-notebook-style LightGBM features, PPG-only features, grouped metrics, and
Optuna model alternatives.

## 1. Prepare data

```bash
uv run python scripts/download_llamac.py --prepare
```

Expected generated inputs:

- `data/raw/llamac_figshare_manifest.json`
- `data/processed/dataset_index.csv`
- `data/extracted/<participant>/answer.csv`
- `data/extracted/<participant>/band_*.csv`
- `data/extracted/<participant>/eeg_*.csv`
- `data/extracted/<participant>/respiration_*.csv`

These files are ignored by git.

## 2. Install modeling dependencies

```bash
uv sync --group ml
```

Use `--group dnn` only for later waveform/image neural-network experiments.

## 3. Build official-notebook-style features

All-channel feature table:

```bash
uv run --group ml python scripts/build_features.py \
  --mode all \
  --workers 4 \
  --output data/processed/features_all.parquet
```

PPG-only feature table:

```bash
uv run --group ml python scripts/build_features.py \
  --mode ppg \
  --workers 4 \
  --output data/processed/features_ppg.parquet
```

Rich PPG-only feature table for tuned alternatives:

```bash
uv run --group ml python scripts/build_features.py \
  --mode ppg_rich \
  --workers 4 \
  --output data/processed/features_ppg_rich.parquet
```

Smoke-test with only two participants:

```bash
uv run --group ml python scripts/build_features.py \
  --mode ppg \
  --limit-subjects 2 \
  --output data/processed/smoke_features_ppg.parquet
```

## 4. Reproduce LightGBM baselines

Default split policy is participant-grouped CV to avoid subject leakage. The
primary public target is the self-reported emotion label (`EmotType`, exposed as
`reported`).

All-channel LightGBM baseline:

```bash
uv run --group ml python scripts/train_model.py \
  --features data/processed/features_all.parquet \
  --model lightgbm \
  --feature-set all \
  --target reported \
  --split grouped \
  --folds 5 \
  --device auto
```

PPG-only LightGBM baseline:

```bash
uv run --group ml python scripts/train_model.py \
  --features data/processed/features_ppg.parquet \
  --model lightgbm \
  --feature-set ppg \
  --target reported \
  --split grouped \
  --folds 5 \
  --device auto
```

To mimic the official notebook more closely, use the notebook's invalid
subject/trial exclusions and stratified CV:

```bash
uv run --group ml python scripts/train_model.py \
  --features data/processed/features_all.parquet \
  --model lightgbm \
  --feature-set all \
  --target reported \
  --split stratified \
  --official-exclusions \
  --folds 5 \
  --device cpu
```

Each result JSON contains:

- top-1 / top-2 / top-3 accuracy,
- macro F1,
- weighted F1,
- balanced accuracy,
- Cohen's kappa,
- confusion matrix,
- per-class precision/recall/F1,
- split metadata,
- feature-file checksum,
- git commit,
- selected backend and GPU fallback reason when relevant.

## 5. Tune alternative model families with Optuna

Example PPG-only search across several non-DNN families:

```bash
uv run --group ml python scripts/tune_models.py \
  --features data/processed/features_ppg.parquet \
  --feature-set ppg \
  --target reported \
  --split grouped \
  --folds 5 \
  --models lightgbm extra_trees hist_gradient_boosting random_forest \
  --trials 30 \
  --metric macro_f1 \
  --device auto
```

Optuna studies are stored under ignored `artifacts/optuna/` SQLite files by
default. Locked best-trial metrics are written to
`artifacts/optuna/locked-results/`.

## 6. Summarize results

```bash
uv run python scripts/summarize_results.py artifacts/results artifacts/optuna/locked-results \
  --output artifacts/results/summary.csv
```

## 7. Validation commands

```bash
uv run python scripts/download_llamac.py --manifest-only
uv run --group dev --group ml pytest
uv run --group ml python scripts/build_features.py --mode ppg --limit-subjects 2 --output data/processed/smoke_features_ppg.parquet
uv run --group ml python scripts/train_model.py --features data/processed/smoke_features_ppg.parquet --model lightgbm --feature-set ppg --target reported --split stratified --folds 2 --device cpu --output artifacts/smoke/lightgbm_ppg_smoke.json
git diff --check
```
