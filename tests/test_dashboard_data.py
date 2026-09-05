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


def test_model_display_names_use_selected_family_from_metadata():
    models = [
        {
            "model_version": "legacy",
            "status": "candidate",
            "model_name": "Xgboost Rul Regressor",
            "evaluated_at": "2026-09-01T00:00:00Z",
            "training_metadata": {"generation": "1.0", "selected_family": "xgboost"},
        },
    ]

    assert model_display_names(models) == {"legacy": "Model 1.0 — XGBoost"}


def test_selectable_models_prefers_metadata_selected_family_over_legacy_duplicate():
    models = [
        {"model_version": "matr-rul-xgboost-1.1-legacy", "status": "candidate", "model_name": "Xgboost Rul Regressor", "evaluated_at": "2026-09-02T00:00:00Z", "training_metadata": {"generation": "1.1"}},
        {"model_version": "canonical-xgb", "status": "candidate", "model_name": "xgboost", "evaluated_at": "2026-09-03T00:00:00Z", "training_metadata": {"generation": "1.1", "selected_family": "xgboost"}},
        {"model_version": "candidate-mlp", "status": "candidate", "model_name": "mlp", "evaluated_at": "2026-09-02T00:00:00Z", "training_metadata": {"generation": "1.2", "selected_family": "mlp"}},
    ]

    assert [model["model_version"] for model in selectable_models(models)] == ["canonical-xgb", "candidate-mlp"]


def test_selectable_models_uses_canonical_generation_rows_only():
    models = [
        {"model_version": "matr-rul-xgboost-1.0-legacy", "status": "candidate", "model_name": "Xgboost Rul Regressor", "evaluated_at": "2026-09-01T00:00:00Z", "training_metadata": {"generation": "1.0"}},
        {"model_version": "model-1.1", "status": "candidate", "model_name": "xgboost", "evaluated_at": "2026-09-02T00:00:00Z", "training_metadata": {"generation": "1.1", "selected_family": "xgboost"}},
        {"model_version": "model-1.2", "status": "candidate", "model_name": "mlp", "evaluated_at": "2026-09-03T00:00:00Z", "training_metadata": {"generation": "1.2", "selected_family": "mlp"}},
        {"model_version": "model-1.3", "status": "candidate", "model_name": "xgboost", "evaluated_at": "2026-09-04T00:00:00Z", "training_metadata": {"generation": "1.3", "selected_family": "xgboost"}},
    ]

    assert [model["model_version"] for model in selectable_models(models)] == ["model-1.1", "model-1.2", "model-1.3"]


def test_selectable_models_prefers_shared_progressive_arrival_v3():
    models = [
        {"model_version": "legacy", "status": "candidate", "model_name": "xgboost", "evaluated_at": "2026-09-01", "training_metadata": {"generation": "1.0", "selected_family": "xgboost"}},
        {"model_version": "state-v2", "status": "candidate", "model_name": "xgboost", "evaluated_at": "2026-09-02", "training_metadata": {"generation": "1.0", "selected_family": "xgboost", "generation_semantics_version": "shared-stream-state-v2"}},
        {"model_version": "arrival-v3", "status": "candidate", "model_name": "mlp", "evaluated_at": "2026-09-03", "training_metadata": {"generation": "1.0", "selected_family": "mlp", "record_class": "canonical_shared_progressive_arrival_v3"}},
    ]

    assert [model["model_version"] for model in selectable_models(models)] == ["arrival-v3"]


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


def test_selectable_models_exclude_retired_and_best_fixed_test_mae_is_default():
    models = [
        {"model_version": "legacy", "status": "retired", "training_metadata": {"generation": 99}},
        {"model_version": "v1", "status": "candidate", "training_metadata": {"generation": 1}, "metrics": {"test": {"mae": 50.15}}},
        {"model_version": "v2", "status": "candidate", "training_metadata": {"generation": "1.2"}, "metrics": {"test": {"mae": 42.53}}},
        {"model_version": "v3", "status": "candidate", "training_metadata": {"generation": 3}, "metrics": {"test": {"mae": 47.49}}},
    ]
    assert [model["model_version"] for model in selectable_models(models)] == ["v1", "v2", "v3"]
    assert latest_model_version(models) == "v2"


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


def test_survival_metrics_stay_separate_from_rul_metrics():
    from src.dashboard_data import survival_model_metrics

    result = survival_model_metrics({"model_version": "s1", "model_name": "cox", "status": "candidate", "evaluated_at": "now", "generation": "1.0", "metrics": {"validation": {"integrated_brier_score": 0.12, "ipcw_c_index": 0.8}, "test": {"integrated_brier_score": 0.14, "ipcw_c_index": 0.75}}, "training_metadata": {"training_battery_count": 26}})
    assert result["Validation IBS"] == 0.12
    assert result["Test IPCW C-index"] == 0.75
    assert "Test MAE" not in result


def test_survival_family_validation_rows_show_both_families_and_selected_winner():
    from src.dashboard_data import survival_family_validation_rows

    rows = survival_family_validation_rows({
        "model_name": "random_survival_forest",
        "metrics": {
            "selection": {"selected_family": "random_survival_forest", "selected_config_id": "trees-200-depth-8"},
            "family_validation": {
                "cox": {"config_id": "alpha-1", "validation": {"integrated_brier_score": 0.02, "ipcw_c_index": 0.8}},
                "random_survival_forest": {"config_id": "trees-200-depth-8", "validation": {"integrated_brier_score": 0.01, "ipcw_c_index": 0.85}},
            },
            "test": {"integrated_brier_score": 0.015, "ipcw_c_index": 0.84, "horizon_brier": {"50": 0.005, "100": 0.01, "200": 0.03}},
        },
    })

    assert rows == [
        {"Model family": "Cox", "Validation IBS": 0.02, "Validation IPCW C-index": 0.8, "Configuration": "alpha-1", "Selected": False},
        {"Model family": "Random Survival Forest", "Validation IBS": 0.01, "Validation IPCW C-index": 0.85, "Configuration": "trees-200-depth-8", "Selected": True},
    ]


def test_performance_gradient_direction_and_scope():
    from src.dashboard_data import performance_gradient
    frame = __import__("pandas").DataFrame({"Name": ["best", "worst"], "Error": [1.0, 3.0], "Score": [0.9, 0.2]})
    styled = performance_gradient(frame, lower_is_better=["Error"], higher_is_better=["Score"])
    styled._compute()
    error_colors = [styled.ctx[(row, 1)][0][1] for row in range(2)]
    score_colors = [styled.ctx[(row, 2)][0][1] for row in range(2)]
    assert error_colors == ["#1e3a8a", "#eff6ff"]
    assert score_colors == ["#1e3a8a", "#eff6ff"]
    assert styled.ctx.get((0, 0), []) == []


def test_performance_gradient_handles_ties_missing_and_single_rows():
    from src.dashboard_data import performance_gradient
    frame = __import__("pandas").DataFrame({"Metric": [1.0, 1.0, None]})
    styled = performance_gradient(frame, lower_is_better=["Metric"])
    styled._compute()
    assert styled.ctx[(0, 0)][0][1] == styled.ctx[(1, 0)][0][1]
    assert styled.ctx.get((2, 0), []) == []
