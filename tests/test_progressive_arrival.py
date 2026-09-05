from datetime import datetime, timezone

import pyarrow.parquet as pq

from src.progressive_arrival import (
    ARRIVAL_EPOCH,
    ARRIVAL_GAP_DAYS,
    GENERATION_ARRIVAL_COUNTS,
    dry_run,
    schedule_manifest,
    snapshot_cutoff,
)


def _manifest():
    return [
        {"battery_id": f"train-{index}", "split": "train", "arrival_rank": index,
         "start_time": "2010-01-01T00:00:00+00:00", "first_source_cycle": 1,
         "last_source_cycle": 100, "eol_cycle": 10, "valid_eol_label": index % 2 == 0,
         "schedule_fingerprint": "original"}
        for index in range(94)
    ]


def test_schedule_uses_only_train_order_epoch_and_fixed_cadence():
    first, registry = schedule_manifest(_manifest())
    altered = [{**row, "valid_eol_label": not row["valid_eol_label"], "eol_cycle": 99} for row in _manifest()]
    second, second_registry = schedule_manifest(altered)

    assert registry == second_registry
    assert [row["start_time"] for row in first] == [row["start_time"] for row in second]
    assert first[0]["start_time"] == ARRIVAL_EPOCH.isoformat()
    assert snapshot_cutoff("1.0", first) == ARRIVAL_EPOCH.replace(tzinfo=timezone.utc) + __import__("datetime").timedelta(days=25 * ARRIVAL_GAP_DAYS)


def test_current_manifest_dry_run_matches_predeclared_progressive_cohorts():
    manifest = pq.read_table("data/processed/matr/arrival_manifest.parquet").to_pylist()
    lifecycle = pq.read_table("data/processed/matr/replay_lifecycle_state").to_pylist()
    _, result = dry_run(manifest, lifecycle)

    assert {generation: row["arrived_train_battery_count"] for generation, row in result.items()} == GENERATION_ARRIVAL_COUNTS
    assert result == {
        "1.0": {"cutoff": "2024-02-09T00:00:00+00:00", "arrived_train_battery_count": 26, "not_yet_arrived_train_battery_count": 75, "observed_eol_train_battery_count": 13, "censored_train_battery_count": 13},
        "1.1": {"cutoff": "2028-03-19T00:00:00+00:00", "arrived_train_battery_count": 51, "not_yet_arrived_train_battery_count": 50, "observed_eol_train_battery_count": 38, "censored_train_battery_count": 13},
        "1.2": {"cutoff": "2032-04-27T00:00:00+00:00", "arrived_train_battery_count": 76, "not_yet_arrived_train_battery_count": 25, "observed_eol_train_battery_count": 59, "censored_train_battery_count": 17},
        "1.3": {"cutoff": "2035-04-12T00:00:00+00:00", "arrived_train_battery_count": 94, "not_yet_arrived_train_battery_count": 7, "observed_eol_train_battery_count": 74, "censored_train_battery_count": 20},
    }
