"""CLI for the official BatteryLife MATR processed archive."""

import argparse
import zipfile
from pathlib import Path

try:
    from .matr_data import normalize_archive
except ImportError:
    from matr_data import normalize_archive


def extract_archive(archive):
    with zipfile.ZipFile(archive) as source:
        destination = archive.parent.resolve()
        for member in source.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive contains a path outside its destination')
        source.extractall(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/batterylife/MATR.zip"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/batterylife/Life labels/" "MATR_labels.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/matr"))
    args = parser.parse_args()
    raw_dir = args.archive.with_suffix("")
    if not raw_dir.exists():
        extract_archive(args.archive)
    print(normalize_archive(raw_dir, args.labels, args.output))


if __name__ == "__main__":
    main()
