# llamac_research

Utilities and notes for research using the LLaMAC dataset for affective computing / emotion prediction.

## Dataset

- Dataset: **LLaMAC: Low-cost Biosignal Sensor based Large Multimodal Dataset for Affective Computing**
- Figshare: <https://figshare.com/articles/dataset/LLaMAC_Low-cost_Biosignal_Sensor_based_Large_Multimodal_Dataset_for_Affective_Computing/28748696/6>
- DOI: `10.6084/m9.figshare.28748696.v6`
- Reference paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12678757/>
- License: CC BY 4.0

The dataset contains low-cost biosignal recordings and questionnaire labels for affective computing, including continuous affect dimensions such as valence, arousal, and dominance, and discrete emotion labels.

## Repository layout

```text
llamac_research/
├── README.md
├── scripts/
│   └── download_llamac.py
├── data/
│   ├── raw/          # downloaded Figshare files; ignored by git
│   ├── extracted/    # optional extracted zip files; ignored by git
│   └── processed/    # derived data; ignored by git by default
└── notebooks/        # analysis notebooks
```

## Environment

This project uses `uv` and pins Python 3.12 via `.python-version` and `pyproject.toml`.

Create or update the project environment:

```bash
uv sync
```

Run Python commands inside the fixed environment:

```bash
uv run python --version
```

Expected Python version:

```text
Python 3.12.x
```

For later modeling work, optional ML dependencies can be installed with:

```bash
uv sync --group ml
```

## Downloading and preparing the dataset

The downloader uses the Figshare API and Python standard library only. It writes a reproducible manifest, verifies file size and MD5 checksums by default, and can prepare extracted files for analysis.

From the repository root, run the full download + analysis preparation:

```bash
uv run python scripts/download_llamac.py --prepare
```

This downloads all Figshare v6 files into `data/raw/`, extracts participant zip files into `data/extracted/`, and creates:

```text
data/processed/dataset_index.csv
```

Expected total download size is about 3.1 GB. Extracted files require additional disk space.

### Smoke test: download only metadata

```bash
uv run python scripts/download_llamac.py --manifest-only
```

This creates:

```text
data/raw/llamac_figshare_manifest.json
```

### Smoke test: download a small subset

Download the first three files selected by natural filename order:

```bash
uv run python scripts/download_llamac.py --limit 3
```

Download exact files:

```bash
uv run python scripts/download_llamac.py --name 1.zip --name 2025_Kitech_Emotion_Data_Code.ipynb
```

Download by regex:

```bash
uv run python scripts/download_llamac.py --pattern '\.ipynb$|\.py$'
```

### Parallelism

The default is 4 parallel downloads:

```bash
uv run python scripts/download_llamac.py --workers 4
```

Use fewer workers on unstable networks:

```bash
uv run python scripts/download_llamac.py --workers 1
```

Use more workers if the connection is stable:

```bash
uv run python scripts/download_llamac.py --workers 8
```

### Resume behavior

The script is safe to re-run. Existing files are skipped when their size and MD5 match the Figshare metadata.

```bash
uv run python scripts/download_llamac.py
```

To force re-download:

```bash
uv run python scripts/download_llamac.py --force
```

### Extract and prepare later

If the raw zip files are already downloaded, rerun the script with `--prepare`. Completed files are skipped, then the archives are extracted and indexed:

```bash
uv run python scripts/download_llamac.py --prepare
```

To force re-extraction:

```bash
uv run python scripts/download_llamac.py --prepare --force-extract
```

## EDA notebook

Launch JupyterLab through `uv` so it uses the project environment:

```bash
uv run jupyter lab
```

Open the starter notebook:

```text
notebooks/01_llamac_eda.ipynb
```

The notebook kernelspec is fixed to:

```text
Python 3.12 (llamac-research)
```

The notebook expects `data/processed/dataset_index.csv`, which is created by `uv run python scripts/download_llamac.py --prepare`.

## Data citation

If you use this dataset, cite the original LLaMAC dataset and paper. Suggested dataset citation information is available from the Figshare DOI page:

```text
10.6084/m9.figshare.28748696.v6
```

## Notes

- Raw and processed data directories are intentionally ignored by git.
- The downloader stores Figshare metadata in `data/raw/llamac_figshare_manifest.json` for reproducibility.
- This repository contains helper code only; it does not redistribute the dataset files.
