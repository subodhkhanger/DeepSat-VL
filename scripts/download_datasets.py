#!/usr/bin/env python3
"""
Flexible dataset downloader for DeepSat-VL.

Supports downloading via:
- Kaggle Datasets API (requires credentials)
- Google Drive (via gdown)
- Direct HTTP/HTTPS URLs

Example usages:
  # Kaggle (requires KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)
  python scripts/download_datasets.py --dataset dior --source kaggle \
      --kaggle-slug <owner/dataset> --out data/DIOR

  # Google Drive file or folder (provide the id)
  python scripts/download_datasets.py --dataset dota --source gdrive \
      --gdrive-id <FILE_OR_FOLDER_ID> --out data/DOTA

  # Direct URL (zip/tar.gz)
  python scripts/download_datasets.py --dataset dior --source http \
      --url https://example.com/DIOR.zip --out data/DIOR

The script will attempt to unzip/untar archives into the target folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def log(msg: str) -> None:
    print(f"[download] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_kaggle_credentials() -> Path:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        return kaggle_json

    # Try environment variables
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        ensure_dir(kaggle_dir)
        kaggle_json.write_text(f'{{"username":"{user}","key":"{key}"}}', encoding="utf-8")
        kaggle_json.chmod(0o600)
        log(f"Created {kaggle_json} from environment variables.")
        return kaggle_json

    raise RuntimeError(
        "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY env vars, "
        "or place kaggle.json under ~/.kaggle."
    )


def download_via_kaggle(slug: str, out_dir: Path) -> Path:
    """Download a Kaggle dataset via the Kaggle API.
    slug format: "owner/dataset".
    Returns the extraction/target directory.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "The 'kaggle' package is required. Install with: pip install kaggle"
        ) from e

    ensure_kaggle_credentials()
    ensure_dir(out_dir)

    api = KaggleApi()
    api.authenticate()

    log(f"Downloading Kaggle dataset '{slug}' to '{out_dir}' (unzip=True)...")
    api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
    log("Kaggle download complete.")
    return out_dir


def _run(cmd: list[str]) -> None:
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def gdown_available() -> bool:
    try:
        import gdown  # noqa: F401
        return True
    except Exception:
        return False


def download_via_gdrive(file_or_folder_id: str, out_dir: Path) -> Path:
    """Download from Google Drive via gdown (file or folder id)."""
    ensure_dir(out_dir)
    if not gdown_available():
        raise RuntimeError("The 'gdown' package is required. Install with: pip install gdown")

    import gdown

    # Detect if folder by trying folder download; fallback to file
    try:
        log(f"Attempting to download GDrive folder id '{file_or_folder_id}'...")
        gdown.download_folder(id=file_or_folder_id, output=str(out_dir), quiet=False, use_cookies=False)
        log("GDrive folder download complete.")
        return out_dir
    except Exception:
        log("Folder download failed or not a folder; attempting single file download...")
        url = f"https://drive.google.com/uc?id={file_or_folder_id}"
        gdown.download(url, output=str(out_dir / "download"), quiet=False)
        log("GDrive file download complete.")
        return out_dir


def download_via_http(url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    target = out_dir / Path(url.split("?")[0]).name

    # Prefer curl if available for robust resuming
    curl = which("curl")
    if curl:
        _run([curl, "-L", "-o", str(target), url])
    else:
        # Fallback to Python requests
        try:
            import requests
        except Exception as e:
            raise RuntimeError("Install 'requests' or have 'curl' available for HTTP download.") from e
        log(f"Downloading via HTTP to {target} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    log(f"HTTP download complete: {target}")
    return out_dir


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(
        name.endswith(ext)
        for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".7z")
    )


def find_archives(base: Path) -> list[Path]:
    return [p for p in base.rglob("*") if p.is_file() and is_archive(p)]


def extract_archive(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    name = archive.name.lower()
    log(f"Extracting {archive} -> {dest}")
    if name.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        import tarfile

        mode = "r"
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            mode = "r:gz"
        elif name.endswith(".tar.bz2") or name.endswith(".tbz2"):
            mode = "r:bz2"
        elif name.endswith(".tar.xz") or name.endswith(".txz"):
            mode = "r:xz"
        with tarfile.open(archive, mode) as tf:
            tf.extractall(dest)
    elif name.endswith(".7z"):
        try:
            import py7zr  # type: ignore
        except Exception as e:
            raise RuntimeError(".7z archive requires 'py7zr' (pip install py7zr)") from e
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=dest)
    else:
        log(f"Unknown archive format for {archive}")


def extract_all_archives(in_dir: Path, out_dir: Path) -> None:
    archives = find_archives(in_dir)
    if not archives:
        log("No archives found to extract.")
        return
    for arc in archives:
        extract_archive(arc, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download datasets (Kaggle/GDrive/HTTP)")
    p.add_argument("--dataset", required=True, help="Dataset name label (e.g., dior, dota)")
    p.add_argument("--source", required=True, choices=["kaggle", "gdrive", "http"], help="Download source")
    p.add_argument("--out", default=None, help="Output directory (default: data/<dataset>)")

    # Kaggle specific
    p.add_argument("--kaggle-slug", default=None, help="Kaggle dataset slug 'owner/dataset'")

    # Google Drive specific
    p.add_argument("--gdrive-id", default=None, help="Google Drive file or folder id")

    # HTTP specific
    p.add_argument("--url", default=None, help="Direct download URL (zip/tar acceptable)")

    # Extraction
    p.add_argument("--no-extract", action="store_true", help="Do not extract archives")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.strip().lower()
    out_dir = Path(args.out) if args.out else Path("data") / dataset.upper()
    ensure_dir(out_dir)

    log(f"Dataset: {dataset}")
    log(f"Source  : {args.source}")
    log(f"Out dir : {out_dir}")

    if args.source == "kaggle":
        if not args.kaggle_slug:
            log("Error: --kaggle-slug is required for Kaggle downloads.")
            log("Find a dataset slug on Kaggle, e.g., 'owner/dataset', and pass it via --kaggle-slug.")
            return 2
        download_via_kaggle(args.kaggle_slug, out_dir)

    elif args.source == "gdrive":
        if not args.gdrive_id:
            log("Error: --gdrive-id is required for Google Drive downloads.")
            return 2
        download_via_gdrive(args.gdrive_id, out_dir)

    elif args.source == "http":
        if not args.url:
            log("Error: --url is required for HTTP downloads.")
            return 2
        download_via_http(args.url, out_dir)

    # Extraction step: search within out_dir for any archives and extract them.
    if not args.no_extract:
        try:
            extract_all_archives(out_dir, out_dir)
        except Exception as e:
            log(f"Extraction warning: {e}")

    log("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

