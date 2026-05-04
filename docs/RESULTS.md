# Current LLaMAC modeling results

Generated locally on 2026-05-05 from LLaMAC Figshare v6 prepared under ignored `data/` paths.
Raw result JSON files and Optuna SQLite studies are intentionally ignored under `artifacts/`.

## Interpretation caveats

- **Grouped CV is the primary public benchmark** because it prevents participant leakage.
- **Official-style stratified CV** reproduces the source notebook shape more closely, including its invalid subject/trial exclusions, but it is not participant-grouped and can produce much higher scores.
- The primary target is self-reported emotion (`EmotType`, shown as `reported`). Intended stimulus labels are included only to compare with the paper/notebook protocol.
- Current PPG-only SOTA within this repo's tuned tabular suite is **logistic_regression** with macro F1 **0.2327** and top-1 **0.2615** on grouped reported-emotion CV.

## Paper-style and baseline LightGBM checks

| Run | Model | Features | Target | Split | Rows | Features n | Top-1 | Top-2 | Top-3 | Macro F1 | Balanced acc | Kappa |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | lightgbm | all | intended | stratified + official exclusions | 4938 | 564 | 0.8445 | 0.9514 | 0.9785 | 0.8435 | 0.8445 | 0.8056 |
| baseline | lightgbm | all | reported | grouped | 5400 | 564 | 0.3167 | 0.5196 | 0.7019 | 0.2725 | 0.2766 | 0.1177 |
| baseline | lightgbm | all | reported | stratified + official exclusions | 4938 | 564 | 0.6887 | 0.8730 | 0.9433 | 0.6901 | 0.6779 | 0.5991 |
| baseline | lightgbm | ppg | intended | stratified + official exclusions | 4938 | 24 | 0.3100 | 0.5492 | 0.7262 | 0.3103 | 0.3100 | 0.1376 |
| baseline | lightgbm | ppg | reported | stratified + official exclusions | 4938 | 24 | 0.3111 | 0.5496 | 0.7329 | 0.3000 | 0.2994 | 0.1213 |

## PPG-only grouped model comparison

| Run | Model | Features | Target | Split | Rows | Features n | Top-1 | Top-2 | Top-3 | Macro F1 | Balanced acc | Kappa |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | svc_rbf | ppg | reported | grouped | 5400 | 24 | 0.2344 | 0.4756 | 0.6470 | 0.2162 | 0.2166 | 0.0300 |
| baseline | lightgbm | ppg | reported | grouped | 5400 | 24 | 0.2226 | 0.4170 | 0.6202 | 0.2100 | 0.2099 | 0.0190 |
| optuna locked | logistic_regression | ppg | reported | grouped | 5400 | 24 | 0.2615 | 0.4456 | 0.6239 | 0.2327 | 0.2422 | 0.0589 |
| optuna locked | svc_rbf | ppg | reported | grouped | 5400 | 24 | 0.2474 | 0.4872 | 0.6643 | 0.2249 | 0.2353 | 0.0497 |
| optuna locked | random_forest | ppg | reported | grouped | 5400 | 24 | 0.2487 | 0.4494 | 0.6191 | 0.2194 | 0.2218 | 0.0381 |
| optuna locked | lightgbm | ppg | reported | grouped | 5400 | 24 | 0.2294 | 0.4278 | 0.6183 | 0.2184 | 0.2184 | 0.0301 |
| optuna locked | extra_trees | ppg | reported | grouped | 5400 | 24 | 0.2543 | 0.4446 | 0.6257 | 0.2123 | 0.2171 | 0.0319 |
| optuna locked | hist_gradient_boosting | ppg | reported | grouped | 5400 | 24 | 0.2622 | 0.4633 | 0.6431 | 0.2103 | 0.2182 | 0.0298 |

## Reproduction commands used

```bash
uv run --group ml python scripts/build_features.py --mode ppg --workers 8 --output data/processed/features_ppg.parquet
uv run --group ml python scripts/build_features.py --mode all --workers 8 --output data/processed/features_all.parquet
uv run --group ml python scripts/train_model.py --features data/processed/features_all.parquet --model lightgbm --feature-set all --target reported --split grouped --folds 5 --device cpu --output artifacts/results/lightgbm_all_reported_grouped.json
uv run --group ml python scripts/train_model.py --features data/processed/features_all.parquet --model lightgbm --feature-set all --target reported --split stratified --official-exclusions --folds 5 --device cpu --output artifacts/results/lightgbm_all_reported_official_stratified.json
uv run --group ml python scripts/train_model.py --features data/processed/features_all.parquet --model lightgbm --feature-set all --target intended --split stratified --official-exclusions --folds 5 --device cpu --output artifacts/results/lightgbm_all_intended_official_stratified.json
uv run --group ml python scripts/train_model.py --features data/processed/features_ppg.parquet --model lightgbm --feature-set ppg --target reported --split grouped --folds 5 --device auto --output artifacts/results/lightgbm_ppg_reported_grouped.json
uv run --group ml python scripts/train_model.py --features data/processed/features_ppg.parquet --model lightgbm --feature-set ppg --target reported --split stratified --official-exclusions --folds 5 --device cpu --output artifacts/results/lightgbm_ppg_reported_official_stratified.json
uv run --group ml python scripts/tune_models.py --features data/processed/features_ppg.parquet --feature-set ppg --target reported --split grouped --folds 5 --models lightgbm extra_trees hist_gradient_boosting random_forest logistic_regression --trials 20 --metric macro_f1 --device cpu --output-dir artifacts/optuna/ppg_reported_grouped
uv run --group ml python scripts/tune_models.py --features data/processed/features_ppg.parquet --feature-set ppg --target reported --split grouped --folds 5 --models svc_rbf --trials 10 --metric macro_f1 --device cpu --output-dir artifacts/optuna/ppg_reported_grouped
uv run python scripts/summarize_results.py artifacts/results artifacts/optuna/ppg_reported_grouped/locked-results --output artifacts/results/summary.csv
```

## Next research extensions

- Add morphology/HRV-specific PPG features beyond the official notebook's PPG summaries.
- Add leakage stress tests grouped by stimulus/video identity.
- Add time-series and PPG-as-image pipelines after the tabular baseline is stable.
