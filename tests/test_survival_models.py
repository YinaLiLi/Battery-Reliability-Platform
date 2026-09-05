from datetime import datetime, timezone

import pytest

from src import survival_models as survival


def _manifest():
    return [
        {"battery_id": "event", "lineage_group_id": "l1", "split": "train", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 3, "eol_cycle": 3, "valid_eol_label": True, "arrival_rank": 0, "schedule_fingerprint": "schedule"},
        {"battery_id": "active", "lineage_group_id": "l2", "split": "train", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 8, "eol_cycle": 8, "valid_eol_label": True, "arrival_rank": 1, "schedule_fingerprint": "schedule"},
        {"battery_id": "censored", "lineage_group_id": "l3", "split": "validation", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 4, "eol_cycle": 8, "valid_eol_label": False, "arrival_rank": 2, "schedule_fingerprint": "schedule"},
        {"battery_id": "future", "lineage_group_id": "l4", "split": "test", "start_time": "2020-01-10T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 3, "eol_cycle": 3, "valid_eol_label": True, "arrival_rank": 3, "schedule_fingerprint": "schedule"},
    ]


def _features():
    return [
        {"dataset": "MATR", "battery_id": battery, "cycle_index": cycle, "feature": float(cycle)}
        for battery, cycles in {"event": range(1, 4), "active": range(1, 6), "censored": range(1, 5), "future": range(1, 4)}.items()
        for cycle in cycles
    ]


def test_landmarks_use_observed_eol_and_right_censoring_at_the_generation_cutoff():
    rows = survival.landmark_rows(
        _features(),
        _manifest(),
        [{"battery_id": "event", "event_type": "eol_observed"}],
        datetime(2020, 1, 5, tzinfo=timezone.utc),
        feature_columns=["feature"],
    )

    by_battery = {}
    for row in rows:
        by_battery.setdefault(row["battery_id"], []).append(row)

    assert {row["duration_cycles"] for row in by_battery["event"]} == {1, 2}
    assert all(row["event_observed"] for row in by_battery["event"])
    assert {row["duration_cycles"] for row in by_battery["active"]} == {1, 2, 3, 4}
    assert not any(row["event_observed"] for row in by_battery["active"])
    assert {row["duration_cycles"] for row in by_battery["censored"]} == {1, 2, 3}
    assert not any(row["event_observed"] for row in by_battery["censored"])
    assert "future" not in by_battery


def test_landmarks_reject_future_target_columns_and_zero_follow_up():
    with pytest.raises(ValueError, match="future-derived"):
        survival.landmark_rows(_features(), _manifest(), [], datetime(2020, 1, 5, tzinfo=timezone.utc), feature_columns=["eol_cycle"])

    rows = survival.landmark_rows(_features(), _manifest(), [], datetime(2020, 1, 5, tzinfo=timezone.utc), feature_columns=["feature"])
    assert all(row["duration_cycles"] > 0 for row in rows)


def test_training_landmark_sampling_keeps_anchors_without_sampling_validation_rows():
    manifest = [
        {"battery_id": "train", "lineage_group_id": "l1", "split": "train", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 13, "eol_cycle": 13, "valid_eol_label": True, "arrival_rank": 0, "schedule_fingerprint": "schedule"},
        {"battery_id": "validation", "lineage_group_id": "l2", "split": "validation", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 12, "eol_cycle": 20, "valid_eol_label": False, "arrival_rank": 1, "schedule_fingerprint": "schedule"},
    ]
    features = [
        {"dataset": "MATR", "battery_id": battery_id, "cycle_index": cycle, "feature": float(cycle)}
        for battery_id, cycles in {"train": range(1, 14), "validation": range(1, 13)}.items()
        for cycle in cycles
    ]
    rows = survival.landmark_rows(features, manifest, [{"battery_id": "train", "event_type": "eol_observed"}], datetime(2020, 1, 20, tzinfo=timezone.utc), feature_columns=["feature"])
    sampled = survival.training_landmark_rows(rows)

    train_rows = [row for row in sampled if row["battery_id"] == "train"]
    validation_rows = [row for row in sampled if row["battery_id"] == "validation"]
    assert [(row["cycle_index"], row["duration_cycles"], row["event_observed"]) for row in train_rows] == [(1, 12, True), (10, 3, True), (12, 1, True)]
    assert [(row["cycle_index"], row["duration_cycles"], row["event_observed"]) for row in validation_rows] == [(cycle, 12 - cycle, False) for cycle in range(1, 12)]


def test_survival_generation_fingerprint_includes_shared_state_and_landmark_sampling(tmp_path):
    manifest = [{"battery_id": "event", "lineage_group_id": "l1", "split": "train", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 3, "eol_cycle": 3, "valid_eol_label": True, "arrival_rank": 0, "schedule_fingerprint": "schedule"}]
    state = {"state_id": "stream-state-test", "finalized_cycle_boundary_fingerprint": "boundary", "arrived_train_battery_ids": ["event"], "observed_eol_train_battery_ids": ["event"], "censored_train_battery_ids": []}
    plan = survival.survival_generation_plan(manifest, state, "1.0", root=tmp_path)

    assert plan["landmark_sampling"] == survival.LANDMARK_SAMPLING
    assert plan["snapshot_id"] == "stream-state-test"
    assert plan["arrived_train_battery_ids"] == ["event"]


def test_validation_selection_prefers_ibs_then_c_index_then_cox():
    winner = survival.select_validation_winner(
        {
            "cox": {"config_id": "cox-a", "validation": {"integrated_brier_score": 0.20, "ipcw_c_index": 0.70}},
            "random_survival_forest": {"config_id": "rsf-a", "validation": {"integrated_brier_score": 0.20, "ipcw_c_index": 0.75}},
        }
    )
    assert winner["family"] == "random_survival_forest"

    tied = survival.select_validation_winner(
        {
            "cox": {"config_id": "cox-a", "validation": {"integrated_brier_score": 0.20, "ipcw_c_index": 0.75}},
            "random_survival_forest": {"config_id": "rsf-a", "validation": {"integrated_brier_score": 0.20, "ipcw_c_index": 0.75}},
        }
    )
    assert tied["family"] == "cox"


def test_curve_grid_includes_required_horizons_and_is_bounded():
    assert survival.HORIZON_GRID == tuple(range(0, 201, 10))
    assert {50, 100, 200}.issubset(survival.HORIZON_GRID)


def test_prediction_validation_rejects_non_monotone_or_missing_required_horizons():
    rows = [
        {"horizon_cycles": 0, "survival_probability": 1.0},
        {"horizon_cycles": 50, "survival_probability": 0.9},
        {"horizon_cycles": 100, "survival_probability": 0.8},
        {"horizon_cycles": 200, "survival_probability": 0.7},
    ]
    survival.validate_prediction_rows(rows)

    with pytest.raises(ValueError, match="non-increasing"):
        survival.validate_prediction_rows([{**row, "survival_probability": 0.95} if row["horizon_cycles"] == 100 else row for row in rows])
    with pytest.raises(ValueError, match="required horizons"):
        survival.validate_prediction_rows(rows[:-1])


def test_ipcw_metrics_ignore_evaluation_rows_beyond_training_support():
    train = [{"duration_cycles": 10, "event_observed": True}]
    evaluation = [
        {"duration_cycles": 5, "event_observed": False},
        {"duration_cycles": 11, "event_observed": False},
    ]

    assert survival._ipcw_supported_rows(train, evaluation) == evaluation[:1]


def test_ipcw_metrics_exclude_rows_after_censoring_support_ends():
    train = [
        {"duration_cycles": 2, "event_observed": True},
        {"duration_cycles": 3, "event_observed": False},
    ]
    evaluation = [
        {"duration_cycles": 2, "event_observed": False},
        {"duration_cycles": 3, "event_observed": False},
    ]

    assert survival._ipcw_supported_rows(train, evaluation) == evaluation[:1]


def test_shared_receipt_cohort_freezes_training_event_semantics():
    lifecycle = [
        {"battery_id": "observed", "eol_observed": True},
        {"battery_id": "censored", "eol_observed": True},
        {"battery_id": "validation", "eol_observed": True},
    ]
    plan = {
        "arrived_train_battery_ids": ["observed", "censored"],
        "observed_eol_train_battery_ids": ["observed"],
    }

    frozen = survival.state_bound_lifecycle(lifecycle, plan)

    assert {row["battery_id"] for row in frozen if row.get("eol_observed")} == {"observed", "validation"}
