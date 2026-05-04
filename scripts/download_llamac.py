#!/usr/bin/env python3
"""Download the LLaMAC Figshare dataset reproducibly.

Dataset:
  LLaMAC: Low-cost Biosignal Sensor based Large Multimodal Dataset for Affective Computing
  DOI: 10.6084/m9.figshare.28748696.v6
  Figshare article: 28748696, version 6

This script uses only the Python standard library. It fetches Figshare metadata,
writes a manifest, downloads missing files, and verifies file size / MD5 checksums
when available.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_ARTICLE_ID = "28748696"
DEFAULT_VERSION = "6"
DEFAULT_OUT_DIR = Path("data/raw")
API_TEMPLATE = "https://api.figshare.com/v2/articles/{article_id}/versions/{version}"


@dataclass(frozen=True)
class FigshareFile:
    id: int
    name: str
    size: int
    download_url: str
    md5: str | None
    mimetype: str | None

    @classmethod
    def from_api(cls, item: dict) -> "FigshareFile":
        return cls(
            id=int(item["id"]),
            name=str(item["name"]),
            size=int(item["size"]),
            download_url=str(item["download_url"]),
            md5=item.get("computed_md5") or item.get("supplied_md5"),
            mimetype=item.get("mimetype"),
        )


def fetch_metadata(article_id: str, version: str) -> dict:
    url = API_TEMPLATE.format(article_id=article_id, version=version)
    request = urllib.request.Request(url, headers={"User-Agent": "llamac-research-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def safe_name(name: str) -> str:
    # Preserve normal Figshare filenames, but block path traversal.
    name = name.replace("\\", "/").split("/")[-1]
    if not name or name in {".", ".."}:
        raise ValueError(f"Unsafe filename from Figshare: {name!r}")
    return name


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(path: Path, expected_size: int, expected_md5: str | None, verify_md5: bool) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != expected_size:
        return False
    if verify_md5 and expected_md5 and md5sum(path) != expected_md5:
        return False
    return True


def download_one(
    file: FigshareFile,
    out_dir: Path,
    verify_md5: bool,
    retries: int,
    timeout: int,
    force: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / safe_name(file.name)
    temp = target.with_suffix(target.suffix + ".part")

    if not force and is_complete(target, file.size, file.md5, verify_md5):
        return {"name": file.name, "status": "skipped", "path": str(target), "bytes": file.size}

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                file.download_url,
                headers={"User-Agent": "llamac-research-downloader/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response, temp.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            actual_size = temp.stat().st_size
            if actual_size != file.size:
                raise RuntimeError(f"size mismatch: got {actual_size}, expected {file.size}")
            if verify_md5 and file.md5:
                actual_md5 = md5sum(temp)
                if actual_md5 != file.md5:
                    raise RuntimeError(f"md5 mismatch: got {actual_md5}, expected {file.md5}")
            temp.replace(target)
            return {"name": file.name, "status": "downloaded", "path": str(target), "bytes": file.size}
        except Exception as exc:  # noqa: BLE001 - CLI should report all retriable failures.
            if temp.exists():
                temp.unlink(missing_ok=True)
            if attempt >= retries:
                return {"name": file.name, "status": "failed", "error": str(exc), "bytes": file.size}
            wait = min(30, 2 ** attempt)
            print(f"[retry {attempt}/{retries}] {file.name}: {exc}; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise AssertionError("unreachable")


def select_files(files: list[FigshareFile], names: list[str], pattern: str | None, limit: int | None) -> list[FigshareFile]:
    selected = files
    if names:
        wanted = set(names)
        selected = [f for f in selected if f.name in wanted]
        missing = sorted(wanted - {f.name for f in selected})
        if missing:
            raise SystemExit(f"Requested filenames not found in Figshare metadata: {', '.join(missing)}")
    if pattern:
        regex = re.compile(pattern)
        selected = [f for f in selected if regex.search(f.name)]
    selected = sorted(selected, key=lambda f: natural_key(f.name))
    if limit is not None:
        selected = selected[:limit]
    return selected


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def write_manifest(metadata: dict, out_dir: Path, files: list[FigshareFile]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": metadata.get("title"),
        "doi": metadata.get("doi"),
        "article_id": metadata.get("id"),
        "version": metadata.get("version"),
        "published_date": metadata.get("published_date"),
        "license": metadata.get("license"),
        "source_api": API_TEMPLATE.format(article_id=DEFAULT_ARTICLE_ID, version=DEFAULT_VERSION),
        "file_count": len(files),
        "total_size_bytes": sum(f.size for f in files),
        "files": [f.__dict__ for f in files],
    }
    path = out_dir / "llamac_figshare_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def safe_zip_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Return zip members after blocking absolute paths and path traversal."""
    members: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        member_path = Path(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe zip member path: {info.filename}")
        members.append(info)
    return members


def extract_archives(out_dir: Path, files: Iterable[FigshareFile], extract_dir: Path, force: bool = False) -> list[Path]:
    """Extract downloaded participant zip files into one folder per zip stem."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_dirs: list[Path] = []
    for file in files:
        if not file.name.lower().endswith(".zip"):
            continue
        archive = out_dir / safe_name(file.name)
        if not archive.exists():
            print(f"[extract skip] missing {archive}")
            continue
        target = extract_dir / archive.stem
        marker = target / ".extracted_ok"
        if marker.exists() and not force:
            print(f"[extract skipped] {archive.name} -> {target}")
            extracted_dirs.append(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {archive.name} -> {target}")
        with zipfile.ZipFile(archive) as zf:
            safe_zip_members(zf)
            zf.extractall(target)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z"), encoding="utf-8")
        extracted_dirs.append(target)
    return extracted_dirs


def build_dataset_index(extract_dir: Path, index_path: Path) -> Path:
    """Create a compact CSV index of extracted files for analysis notebooks."""
    rows: list[dict[str, str | int]] = []
    participant_dirs = sorted([p for p in extract_dir.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)) if extract_dir.exists() else []
    for participant_dir in participant_dirs:
        for path in sorted(participant_dir.rglob("*"), key=lambda p: natural_key(str(p.relative_to(participant_dir)))):
            if not path.is_file() or path.name == ".extracted_ok":
                continue
            relative_path = path.relative_to(extract_dir)
            rows.append(
                {
                    "participant_id": participant_dir.name,
                    "file_name": path.name,
                    "relative_path": str(relative_path),
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                }
            )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "file_name", "relative_path", "suffix", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[index] {index_path} ({len(rows)} files)")
    return index_path


def prepare_analysis_data(out_dir: Path, selected: list[FigshareFile], extract_dir: Path, force_extract: bool = False) -> Path:
    """Unzip data and write an index consumed by the EDA notebook."""
    extract_archives(out_dir, selected, extract_dir, force=force_extract)
    return build_dataset_index(extract_dir, Path("data/processed/dataset_index.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the LLaMAC Figshare dataset.")
    parser.add_argument("--article-id", default=DEFAULT_ARTICLE_ID, help="Figshare article id. Default: %(default)s")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Figshare version. Default: %(default)s")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Download directory. Default: %(default)s")
    parser.add_argument("--name", action="append", default=[], help="Download one exact filename. Repeatable, e.g. --name 1.zip")
    parser.add_argument("--pattern", help="Regex filter for filenames, e.g. '^(1|2|3)\\.zip$' or '\\.ipynb$'")
    parser.add_argument("--limit", type=int, help="Download only the first N selected files, useful for smoke tests")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads. Default: %(default)s")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds. Default: %(default)s")
    parser.add_argument("--retries", type=int, default=4, help="Retries per file. Default: %(default)s")
    parser.add_argument("--no-md5", action="store_true", help="Skip MD5 verification after download")
    parser.add_argument("--force", action="store_true", help="Re-download even when an existing file passes size/MD5 checks")
    parser.add_argument("--manifest-only", action="store_true", help="Only write metadata manifest; do not download files")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zip files after download")
    parser.add_argument("--prepare", action="store_true", help="After download, extract zip files and build data/processed/dataset_index.csv")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract zip files even when .extracted_ok markers exist")
    parser.add_argument("--extract-dir", type=Path, default=Path("data/extracted"), help="Extraction directory. Default: %(default)s")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = fetch_metadata(args.article_id, args.version)
    files = [FigshareFile.from_api(item) for item in metadata.get("files", []) if item.get("download_url")]
    selected = select_files(files, args.name, args.pattern, args.limit)

    if not selected:
        print("No files selected.", file=sys.stderr)
        return 2

    manifest_path = write_manifest(metadata, args.out_dir, files)
    total_size = sum(f.size for f in selected)
    print(f"Dataset: {metadata.get('title')}")
    print(f"DOI: {metadata.get('doi')}")
    print(f"Manifest: {manifest_path}")
    print(f"Selected files: {len(selected)} / {len(files)} ({total_size:,} bytes)")

    if args.manifest_only:
        return 0

    verify_md5 = not args.no_md5
    results: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(download_one, f, args.out_dir, verify_md5, args.retries, args.timeout, args.force): f
            for f in selected
        }
        for future in futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            status = result["status"]
            name = result["name"]
            if status == "failed":
                print(f"[failed] {name}: {result.get('error')}")
            else:
                print(f"[{status}] {name}")

    failed = [r for r in results if r["status"] == "failed"]
    downloaded = [r for r in results if r["status"] == "downloaded"]
    skipped = [r for r in results if r["status"] == "skipped"]
    print(f"Done: downloaded={len(downloaded)}, skipped={len(skipped)}, failed={len(failed)}")

    if failed:
        (args.out_dir / "failed_downloads.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"Failed list written to {args.out_dir / 'failed_downloads.json'}", file=sys.stderr)
        return 1

    if args.prepare:
        prepare_analysis_data(args.out_dir, selected, args.extract_dir, force_extract=args.force_extract)
    elif args.extract:
        extract_archives(args.out_dir, selected, args.extract_dir, force=args.force_extract)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
