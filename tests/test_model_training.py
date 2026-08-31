from collections import Counter
from datetime import datetime, timedelta

import pytest

from src.model_training import FEATURE_COLUMNS
from src.model_training import DEFAULT_THRESHOLD
from src.model_training import PRIMARY_END
from src.model_training import PRIMARY_START
from src.model_training import TARGET
from src.model_training import vehicle_timing_band
from src.model_training import choose_recall_threshold
from src.model_training import make_primary_splits
from src.model_training import make_temporal_splits
from src.model_training import select_model
from src.model_training import split_summary
from src.model_training import _feature_shift
from src.model_training import _metrics


def _timing_rows():
    bands = (
        ["no_eol"] * 10
        + ["early_positive_window"] * 2
        + ["mid_positive_window"] * 4
        + ["late_positive_window"] * 3
        + ["after_primary_horizon"]
    )
    first_positive = {
        "early_positive_window": datetime(2025, 2, 1),
        "mid_positive_window": datetime(2025, 3, 1),
        "late_positive_window": datetime(2025, 3, 20),
        "after_primary_horizon": datetime(2025, 4, 1),
    }
    rows = []
    for number, band in enumerate(bands):
        for timestamp in (PRIMARY_START, datetime(2025, 3, 31), datetime(2025, 4, 1)):
            rows.append(
                {
                    "vehicle_id": f"EV-{number:04d}",
                    "timestamp": timestamp,
                    "battery_type": "standard" if number % 2 else "long_range",
                    "region": ("south", "west", "midwest")[number % 3],
                    TARGET: int(band != "no_eol" and timestamp >= first_positive.get(band, datetime.max)),
                }
            )
    return rows


def _rows():
    rows = []
    for number in range(12):
        vehicle_id = f"EV-{number:04d}"
        battery_type = "standard"
        region = "south"
        for day in range(4):
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "timestamp": datetime(2025, 3, 1) + timedelta(days=day),
                    "battery_type": battery_type,
                    "region": region,
                    "failure_within_30_operating_days": int(number < 6),
                }
            )
    return rows


def test_primary_splits_use_exact_vehicle_cohorts_and_common_horizon():
    rows = []
    for number in range(20):
        for timestamp in (PRIMARY_START, PRIMARY_END, PRIMARY_END + timedelta(days=1)):
            rows.append(
                {
                    "vehicle_id": f"EV-{number:04d}",
                    "timestamp": timestamp,
                    "battery_type": "standard",
                    "region": "south",
                    TARGET: int(number < 10),
                }
            )

    splits = make_primary_splits(rows, min_class_vehicles=1, min_class_rows=1)
    vehicles = {name: {row["vehicle_id"] for row in rows} for name, rows in splits.items()}
    assert {name: len(value) for name, value in vehicles.items()} == {"train": 12, "validation": 4, "test": 4}
    assert not vehicles["train"] & vehicles["validation"]
    assert not vehicles["train"] & vehicles["test"]
    assert not vehicles["validation"] & vehicles["test"]
    assert all(PRIMARY_START <= row["timestamp"] <= PRIMARY_END for rows in splits.values() for row in rows)


def test_primary_split_uses_deterministic_timing_bands_without_vehicle_overlap():
    rows = _timing_rows()

    first = make_primary_splits(rows, min_class_vehicles=1, min_class_rows=1)
    second = make_primary_splits(rows, min_class_vehicles=1, min_class_rows=1)
    first_vehicles = {name: {row["vehicle_id"] for row in values} for name, values in first.items()}
    second_vehicles = {name: {row["vehicle_id"] for row in values} for name, values in second.items()}

    assert {name: len(value) for name, value in first_vehicles.items()} == {"train": 12, "validation": 4, "test": 4}
    assert first_vehicles == second_vehicles
    assert not first_vehicles["train"] & first_vehicles["validation"]
    assert not first_vehicles["train"] & first_vehicles["test"]
    assert not first_vehicles["validation"] & first_vehicles["test"]
    assert vehicle_timing_band([row for row in rows if row["vehicle_id"] == "EV-0019"]) == "after_primary_horizon"
    total_bands = Counter(vehicle_timing_band([row for row in rows if row["vehicle_id"] == vehicle_id]) for vehicle_id in first_vehicles["train"] | first_vehicles["validation"] | first_vehicles["test"])
    for name, expected_share in (("train", 0.60), ("validation", 0.20), ("test", 0.20)):
        actual_bands = Counter(vehicle_timing_band([row for row in rows if row["vehicle_id"] == vehicle_id]) for vehicle_id in first_vehicles[name])
        assert all(abs(actual_bands[band] - total * expected_share) <= 1 for band, total in total_bands.items())


def test_temporal_splits_allow_vehicle_overlap_by_design():
    splits = make_temporal_splits(
        _rows(),
        train_end=datetime(2025, 3, 1),
        validation_end=datetime(2025, 3, 2),
        test_end=datetime(2025, 3, 4),
        min_class_vehicles=1,
        min_class_rows=1,
    )

    vehicles = {name: {row["vehicle_id"] for row in rows} for name, rows in splits.items()}
    assert vehicles["train"] & vehicles["validation"]
    assert vehicles["validation"] & vehicles["test"]


def test_split_summary_reports_vehicle_rows_and_prevalence():
    summary = split_summary(
        {"train": [{"vehicle_id": "EV-1", TARGET: 0}, {"vehicle_id": "EV-2", TARGET: 1}]}
    )

    assert summary["train"] == {"vehicles": 2, "rows": 2, "positive": 1, "negative": 1, "positive_rate": 0.5}


def test_feature_shift_ignores_null_numeric_values():
    rows = {
        "train": [{column: 1.0 for column in FEATURE_COLUMNS}, {column: 4.0 for column in FEATURE_COLUMNS}],
        "validation": [{column: 2.0 for column in FEATURE_COLUMNS}],
        "test": [{column: 3.0 for column in FEATURE_COLUMNS}],
    }
    rows["train"][0]["previous_pack_voltage"] = None

    assert _feature_shift(rows)["numeric_median"]["previous_pack_voltage"]["train"] == 4.0


def test_feature_columns_exclude_identifiers_timestamps_and_target():
    assert "vehicle_id" not in FEATURE_COLUMNS
    assert "event_id" not in FEATURE_COLUMNS
    assert "timestamp" not in FEATURE_COLUMNS
    assert "failure_within_30_operating_days" not in FEATURE_COLUMNS
    assert "previous_pack_voltage" in FEATURE_COLUMNS


def test_choose_recall_threshold_uses_highest_threshold_meeting_target():
    threshold = choose_recall_threshold([1, 1, 0, 0], [0.9, 0.7, 0.6, 0.1], recall_target=1.0)

    assert threshold == 0.7


def test_metrics_use_fixed_operational_threshold_and_report_alert_rate():
    metrics = _metrics([0, 0, 1, 1], [0.1, 0.5, 0.6, 0.2], DEFAULT_THRESHOLD)

    assert DEFAULT_THRESHOLD == 0.50
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["alert_rate"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_select_model_prefers_the_simpler_model_within_pr_auc_tolerance():
    selected = select_model(
        [
            {"name": "random_forest_depth_16", "validation": {"pr_auc": 0.802}},
            {"name": "logistic_regression", "validation": {"pr_auc": 0.800}},
            {"name": "xgboost_depth_3", "validation": {"pr_auc": 0.804}},
        ]
    )

    assert selected["name"] == "logistic_regression"
