"""CLI for the official BatteryLife MATR processed archive."""

import argparse
import subprocess
from pathlib import Path

from matr_data import normalize_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/batterylife/MATR.zip"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/batterylife/Life labels/" "MATR_labels.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/matr"))
    args = parser.parse_args()
    raw_dir = args.archive.with_suffix("")
    if not raw_dir.exists():
        subprocess.run(["unzip", "-oq", str(args.archive), "-d", str(raw_dir.parent)], check=True)
    print(normalize_archive(raw_dir, args.labels, args.output))


if __name__ == "__main__":
    main()
