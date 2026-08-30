from pathlib import Path

from src import download_data


def test_download_is_idempotent_when_targets_exist(tmp_path, monkeypatch):
    battery_dir = tmp_path / "nasa_batteries"
    battery_dir.mkdir(parents=True, exist_ok=True)

    for battery in download_data.TARGET_MAT_FILES:
        (battery_dir / battery).write_bytes(b"preexisting")

    def unexpected(*_args, **_kwargs):
        raise RuntimeError("download should not run when files exist")

    monkeypatch.setattr(download_data, "_try_download", unexpected)
    monkeypatch.setattr(download_data, "_prune_raw_data_root", unexpected)

    assert download_data._download_nasa_batteries(battery_dir) is True
    assert all(download_data._is_valid_file(battery_dir / target) for target in download_data.TARGET_MAT_FILES)


def test_cleanup_removes_unrelated_nasa_raw_artifacts(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data_raw"
    raw_dir.mkdir()
    battery_dir = raw_dir / "nasa_batteries"
    battery_dir.mkdir()

    for battery in download_data.TARGET_MAT_FILES:
        (battery_dir / battery).write_bytes(b"ok")
    (battery_dir / "B0025.mat").write_bytes(b"extra")
    (battery_dir / "README.txt").write_text("unused")
    (raw_dir / "nasa_battery_data.zip").write_bytes(b"zip")
    extra_dir = raw_dir / "5. Battery Data Set"
    extra_dir.mkdir()
    (extra_dir / "unrelated").write_bytes(b"data")

    monkeypatch.setattr(download_data, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(download_data, "BATTERY_DIR", battery_dir)

    # Keep only calibrated batteries and remove unrelated files/dirs.
    download_data._clean_target_directory(battery_dir)
    download_data._prune_raw_data_root()

    remaining = sorted(p.name for p in battery_dir.iterdir() if p.is_file())
    assert remaining == sorted(download_data.TARGET_MAT_FILES)
    assert not (raw_dir / "nasa_battery_data.zip").exists()
    assert not (raw_dir / "5. Battery Data Set").exists()
