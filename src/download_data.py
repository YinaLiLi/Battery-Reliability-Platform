from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import io
import zipfile

import requests


TARGET_BATTERY_IDS: list[str] = ["B0005", "B0006", "B0007", "B0018"]
TARGET_MAT_FILES: list[str] = [f"{battery}.mat" for battery in TARGET_BATTERY_IDS]

RAW_DATA_DIR = Path("data/raw")
BATTERY_DIR = RAW_DATA_DIR / "nasa_batteries"
OFFICIAL_ARCHIVE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
FY08Q4_INNER_ARCHIVE = "1. BatteryAgingARC-FY08Q4.zip"


def download_file(url: str, output_path: Path):
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)

    print(f"Downloaded to: {output_path}")


def _is_valid_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _missing_targets(directory: Path) -> list[str]:
    return [name for name in TARGET_MAT_FILES if not _is_valid_file(directory / name)]


def _clean_target_directory(directory: Path) -> None:
    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.is_file():
            if item.name not in TARGET_MAT_FILES:
                item.unlink()
        else:
            # Keep only raw .mat files for the MVP fleet simulator calibration.
            import shutil

            shutil.rmtree(item)

    for target in TARGET_MAT_FILES:
        if not _is_valid_file(directory / target):
            (directory / target).unlink(missing_ok=True)


def _prune_raw_data_root() -> None:
    # Remove leftover NASA archives/extra extraction folders while preserving other raw datasets.
    extras = [RAW_DATA_DIR / "nasa_battery_data.zip", RAW_DATA_DIR / "5. Battery Data Set"]
    import shutil

    for extra in extras:
        if extra.is_file():
            extra.unlink()
        elif extra.is_dir():
            shutil.rmtree(extra)


def _try_download(url: str, destination: Path) -> bool:
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        download_file(url, temp_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    if temp_path.stat().st_size == 0:
        temp_path.unlink()
        return False

    temp_path.replace(destination)
    return True


def _extract_from_fy08q4_archive(archive_url: str, directory: Path, missing: list[str]) -> None:
    with TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "nasa_battery_data.zip"
        if not _try_download(archive_url, archive_path):
            return

        with zipfile.ZipFile(archive_path, "r") as outer_archive:
            inner_names = [
                name
                for name in outer_archive.namelist()
                if name.endswith(FY08Q4_INNER_ARCHIVE)
                or name.endswith(f"/{FY08Q4_INNER_ARCHIVE}")
            ]

            if not inner_names:
                return

            inner_bytes = outer_archive.read(inner_names[0])

        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_archive:
            available = inner_archive.namelist()
            for target in missing:
                member = next(
                    (name for name in available if name.endswith(f"/{target}") or name == target),
                    None,
                )
                if not member:
                    continue

                destination = directory / target
                destination.write_bytes(inner_archive.read(member))


def _validate_expected_files(directory: Path) -> bool:
    return all(_is_valid_file(directory / target) for target in TARGET_MAT_FILES)


def _download_nasa_batteries(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)

    missing = _missing_targets(directory)
    if not missing:
        return True

    _extract_from_fy08q4_archive(OFFICIAL_ARCHIVE_URL, directory, missing)

    return _validate_expected_files(directory)


if __name__ == "__main__":
    BATTERY_DIR.mkdir(parents=True, exist_ok=True)
    _download_nasa_batteries(BATTERY_DIR)
    _clean_target_directory(BATTERY_DIR)
    _prune_raw_data_root()
    if not _validate_expected_files(BATTERY_DIR):
        raise RuntimeError(
            "Unable to obtain all required NASA batteries: "
            + ", ".join(TARGET_MAT_FILES)
        )

    print("NASA battery raw files ready:")
    for battery in TARGET_MAT_FILES:
        print(f"- {battery}")
