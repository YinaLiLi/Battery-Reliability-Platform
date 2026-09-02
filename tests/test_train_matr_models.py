import json

import pyarrow as pa
import pyarrow.parquet as pq

from src import train_matr_models as training


def _manifest():
    rows = []
    for index in range(100):
        rows.append(
            {
                "battery_id": f"battery-{index:03}",
                "split": "train" if index < 94 else "validation",
                "valid_eol_label": index < 94 or index == 94,
                "arrival_rank": index,
                "schedule_fingerprint": "schedule-v1",
            }
        )
    return rows


def _events(batteries):
    return [
        {"battery_id": battery, "event_type": event_type}
        for battery in batteries
        for event_type in ("eol_observed", "replay_complete")
    ]


def test_generation_plan_uses_only_completed_valid_train_batteries_in_arrival_order():
    manifest = _manifest()
    plan = training.generation_plan(manifest, _events([f"battery-{index:03}" for index in range(30)]), set())

    assert plan["training_battery_count"] == 26
    assert plan["battery_ids"] == [f"battery-{index:03}" for index in range(26)]
    assert plan["generation"] == "1.0"
    assert plan["model_version"].startswith("matr-rul-model-1.0-")


def test_validation_selection_chooses_the_best_non_xgboost_family_and_breaks_ties_deterministically():
    selected = training.select_validation_winner(
        {
            "ridge": {"config_id": "ridge-a", "validation": {"mae": 10.0, "rmse": 15.0, "r2": 0.7}},
            "random_forest": {"config_id": "rf-a", "validation": {"mae": 10.0, "rmse": 14.0, "r2": 0.7}},
            "xgboost": {"config_id": "xgb-a", "validation": {"mae": 9.0, "rmse": 20.0, "r2": 0.7}},
            "mlp": {"config_id": "mlp-a", "validation": {"mae": 8.0, "rmse": 30.0, "r2": 0.7}},
        }
    )
    assert selected["family"] == "mlp"

    tied = training.select_validation_winner(
        {
            "ridge": {"config_id": "ridge-a", "validation": {"mae": 10.0, "rmse": 15.0, "r2": 0.7}},
            "random_forest": {"config_id": "rf-a", "validation": {"mae": 10.0, "rmse": 15.0, "r2": 0.7}},
            "xgboost": {"config_id": "xgb-a", "validation": {"mae": 11.0, "rmse": 15.0, "r2": 0.7}},
            "mlp": {"config_id": "mlp-a", "validation": {"mae": 12.0, "rmse": 15.0, "r2": 0.7}},
        }
    )
    assert tied["family"] == "ridge"


def test_generation_plan_skips_published_threshold_and_is_fingerprint_idempotent(tmp_path):
    manifest = _manifest()
    events = _events([f"battery-{index:03}" for index in range(60)])
    first = training.generation_plan(manifest, events, {26})
    assert first["training_battery_count"] == 51

    artifact = tmp_path / "model_generations" / first["fingerprint"]
    artifact.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"model_version": first["model_version"], "training_metadata_json": json.dumps(first["metadata"])}]), artifact / "candidate_model_evaluation.parquet")

    assert training.generation_plan(manifest, events, {26}, output_root=tmp_path) is None


def test_fixed_benchmark_ids_exclude_invalid_labels_without_moving_cohorts():
    manifest = [
        {"battery_id": "train", "split": "train", "valid_eol_label": True},
        {"battery_id": "validation-valid", "split": "validation", "valid_eol_label": True},
        {"battery_id": "validation-invalid", "split": "validation", "valid_eol_label": False},
        {"battery_id": "test-valid", "split": "test", "valid_eol_label": True},
        {"battery_id": "test-invalid", "split": "test", "valid_eol_label": False},
    ]

    assert training.fixed_benchmark_ids(manifest) == {
        "validation": {"validation-valid"},
        "test": {"test-valid"},
    }
