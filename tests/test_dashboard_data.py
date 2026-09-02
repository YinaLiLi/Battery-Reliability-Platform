import json

from src.dashboard_data import (
    lifecycle_stage,
    lowest_rows,
    model_display_names,
    model_metrics,
    selectable_models,
    latest_model_version,
    soh_percent,
)


def test_lifecycle_stage_is_unknown_without_a_model_prediction():
    assert lifecycle_stage(120, None) == "Unavailable"


def test_lifecycle_stage_uses_predicted_eol_not_historical_rul():
    assert lifecycle_stage(20, 100) == "Early"
    assert lifecycle_stage(50, 100) == "Mid"
    assert lifecycle_stage(120, 60) == "Late"


def test_model_metrics_flattens_evaluation_and_keeps_missing_training_metadata_unrecorded():
    evaluation = {
        "model_version": "candidate-1",
        "status": "candidate",
        "evaluated_at": "2026-09-01T00:00:00Z",
        "metrics": json.dumps(
            {
                "test": {"mae": 12.5, "rmse": 20.0, "r2": 0.91},
                "validation": {"mae": 14.0},
                "lifecycle_stage_mae": {"early": 18.0, "mid": 12.0, "late": 7.0},
            }
        ),
        "training_metadata": {},
    }

    assert model_metrics(evaluation) == {
        "Model version": "candidate-1",
        "Generation": "Not recorded",
        "Selected model family": "Not Recorded",
        "Training batteries": "Not recorded",
        "Model fingerprint": "Not recorded",
        "Status": "candidate",
        "Validation MAE": 14.0,
        "Evaluated at": "2026-09-01T00:00:00Z",
        "Test MAE": 12.5,
        "Test RMSE": 20.0,
        "Test R²": 0.91,
        "Early MAE": 18.0,
        "Mid MAE": 12.0,
        "Late MAE": 7.0,
        "Training data": "Not recorded",
    }


def test_model_display_names_are_stable_and_hide_internal_ids_from_primary_labels():
    models = [
        {"model_version": "candidate-b", "status": "candidate", "evaluated_at": "2026-09-02"},
        {"model_version": "champion-id", "status": "champion", "evaluated_at": "2026-09-01"},
        {"model_version": "candidate-a", "status": "candidate", "evaluated_at": "2026-09-01"},
    ]
    assert model_display_names(models) == {"champion-id": "XGBoost 1.0", "candidate-a": "XGBoost 1.1", "candidate-b": "XGBoost 1.2"}


def test_model_display_names_use_generation_metadata_and_exclude_retired_models():
    models = [
        {"model_version": "legacy", "status": "retired", "evaluated_at": "2026-09-01", "training_metadata": {"generation": 0}},
        {"model_version": "snapshot-1", "model_name": "mlp", "status": "candidate", "evaluated_at": "2026-09-02", "training_metadata": {"generation": "1.1"}},
        {"model_version": "snapshot-0", "model_name": "ridge", "status": "candidate", "evaluated_at": "2026-09-01", "training_metadata": {"generation": "1.0"}},
    ]

    assert model_display_names(models) == {"snapshot-0": "Model 1.0 — Ridge", "snapshot-1": "Model 1.1 — MLP"}


def test_model_metrics_exposes_generation_selection_and_family_validation_results():
    evaluation = {
        "model_version": "model-1-2",
        "model_name": "random_forest",
        "status": "candidate",
        "evaluated_at": "2026-09-02T00:00:00Z",
        "model_fingerprint": "abc123",
        "generation": "1.2",
        "metrics": {"test": {"mae": 12.0, "rmse": 18.0, "r2": 0.9}, "lifecycle_stage_mae": {}, "family_validation": {"ridge": {"config_id": "a", "validation": {"mae": 13.0, "rmse": 19.0, "r2": 0.8}}}},
        "training_metadata": {"training_battery_count": 76, "selected_family": "random_forest"},
    }

    flattened = model_metrics(evaluation)

    assert flattened["Generation"] == "1.2"
    assert flattened["Selected model family"] == "Random Forest"
    assert flattened["Training batteries"] == 76
    assert flattened["Model fingerprint"] == "abc123"


def test_selectable_models_exclude_retired_and_latest_is_default():
    models = [
        {"model_version": "legacy", "status": "retired", "training_metadata": {"generation": 99}},
        {"model_version": "v1", "status": "candidate", "training_metadata": {"generation": 1}},
        {"model_version": "v3", "status": "candidate", "training_metadata": {"generation": 3}},
        {"model_version": "v2", "status": "candidate", "training_metadata": {"generation": 2}},
    ]
    assert [model["model_version"] for model in selectable_models(models)] == ["v1", "v2", "v3"]
    assert latest_model_version(models) == "v3"


def test_soh_percent_converts_measured_fraction_for_display():
    assert soh_percent(0.7301) == 73.01


def test_lowest_rows_ranks_available_values_without_assigning_risk():
    batteries = [
        {"battery_id": "a", "measured_soh": 0.80},
        {"battery_id": "b", "measured_soh": None},
        {"battery_id": "c", "measured_soh": 0.23},
        {"battery_id": "d", "measured_soh": 0.40},
    ]

    assert [row["battery_id"] for row in lowest_rows(batteries, "measured_soh", limit=2)] == ["c", "d"]
