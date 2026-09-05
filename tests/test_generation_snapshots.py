from datetime import datetime, timezone

from src.generation_snapshots import build_generation_plan, cohort_at_cutoff


def _manifest():
    return [
        {"battery_id": "event", "split": "train", "arrival_rank": 0, "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 10, "eol_cycle": 3, "valid_eol_label": True, "schedule_fingerprint": "schedule"},
        {"battery_id": "active", "split": "train", "arrival_rank": 1, "start_time": "2020-01-02T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 10, "eol_cycle": 10, "valid_eol_label": True, "schedule_fingerprint": "schedule"},
        {"battery_id": "later", "split": "train", "arrival_rank": 2, "start_time": "2020-01-10T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 10, "eol_cycle": 4, "valid_eol_label": True, "schedule_fingerprint": "schedule"},
    ]


def test_shared_snapshot_cohort_partitions_arrived_batteries_by_observed_eol():
    cohort = cohort_at_cutoff(_manifest(), [{"battery_id": "event", "event_type": "eol_observed"}], datetime(2020, 1, 5, tzinfo=timezone.utc))
    assert cohort == {"arrived_train_battery_ids": ["event", "active"], "observed_eol_train_battery_ids": ["event"], "censored_train_battery_ids": ["active"]}


def test_shared_plan_keeps_one_cohort_identity_for_both_model_families(tmp_path):
    state = {"state_id": "state", "finalized_cycle_boundary_fingerprint": "boundary", "arrived_train_battery_ids": ["event", "active"], "observed_eol_train_battery_ids": ["event"], "censored_train_battery_ids": ["active"]}
    first = build_generation_plan("1.0", _manifest(), state, model_config={"family": "rul"}, feature_version="features", artifact_root=tmp_path)
    second = build_generation_plan("1.0", _manifest(), state, model_config={"family": "survival"}, feature_version="features", artifact_root=tmp_path)
    assert first["snapshot_id"] == second["snapshot_id"] == "state"
    assert first["arrived_train_battery_ids"] == second["arrived_train_battery_ids"]
    assert first["observed_eol_train_battery_ids"] == second["observed_eol_train_battery_ids"]
    assert first["fingerprint"] != second["fingerprint"]


def test_streaming_state_cutoff_is_the_shared_generation_cutoff(tmp_path):
    state = {
        "state_id": "state", "finalized_cycle_boundary_fingerprint": "boundary",
        "arrived_train_battery_ids": ["event", "active"],
        "observed_eol_train_battery_ids": ["event"], "censored_train_battery_ids": ["active"],
        "cutoff_metadata": {"replay_cutoff": "2025-01-01T12:00:00+00:00"},
    }

    plan = build_generation_plan("1.0", _manifest(), state, model_config={}, feature_version="features", artifact_root=tmp_path)

    assert plan["cutoff"] == datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
