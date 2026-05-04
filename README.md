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

## Downloading the dataset

The downloader uses the Figshare API and Python standard library only. It writes a reproducible manifest and verifies file size and MD5 checksums by default.

From the repository root:

```bash
python scripts/download_llamac.py
```

This downloads all Figshare v6 files into:

```text
data/raw/
```

Expected total download size is about 3.1 GB.

### Smoke test: download only metadata

```bash
python scripts/download_llamac.py --manifest-only
```

This creates:

```text
data/raw/llamac_figshare_manifest.json
```

### Smoke test: download a small subset

Download the first three files selected by natural filename order:

```bash
python scripts/download_llamac.py --limit 3
```

Download exact files:

```bash
python scripts/download_llamac.py --name 1.zip --name 2025_Kitech_Emotion_Data_Code.ipynb
```

Download by regex:

```bash
python scripts/download_llamac.py --pattern '\.ipynb$|\.py$'
```

### Parallelism

The default is 4 parallel downloads:

```bash
python scripts/download_llamac.py --workers 4
```

Use fewer workers on unstable networks:

```bash
python scripts/download_llamac.py --workers 1
```

Use more workers if the connection is stable:

```bash
python scripts/download_llamac.py --workers 8
```

### Resume behavior

The script is safe to re-run. Existing files are skipped when their size and MD5 match the Figshare metadata.

```bash
python scripts/download_llamac.py
```

To force re-download:

```bash
python scripts/download_llamac.py --force
```

### Extract downloaded zip files

```bash
python scripts/download_llamac.py --extract
```

By default, zip files are extracted into:

```text
data/extracted/
```

You can choose another extraction directory:

```bash
python scripts/download_llamac.py --extract --extract-dir /path/to/extracted_llamac
```

## Data citation

If you use this dataset, cite the original LLaMAC dataset and paper. Suggested dataset citation information is available from the Figshare DOI page:

```text
10.6084/m9.figshare.28748696.v6
```

## Notes

- Raw and processed data directories are intentionally ignored by git.
- The downloader stores Figshare metadata in `data/raw/llamac_figshare_manifest.json` for reproducibility.
- This repository contains helper code only; it does not redistribute the dataset files.
