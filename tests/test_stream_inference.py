import pytest

from src.stream_inference import current_prediction_rows


class Model:
    def predict(self, matrix):
        return [-2.0 if row[0] == 1 else 4.0 for row in matrix]


def test_current_inference_excludes_benchmarks_and_freezes_prior_eol():
    rows = current_prediction_rows(
        Model(),
        [
            {"dataset": "MATR", "battery_id": "serve", "cycle_index": 1, "feature": 1.0, "replay_sequence": 8},
            {"dataset": "MATR", "battery_id": "benchmark", "cycle_index": 1, "feature": 1.0, "replay_sequence": 8},
            {"dataset": "MATR", "battery_id": "frozen", "cycle_index": 3, "feature": 2.0, "replay_sequence": 9},
        ],
        feature_columns=["feature"],
        model_version="model-1",
        model_fingerprint="fingerprint",
        state_id="stream-state-1",
        selection_revision=2,
        benchmark_battery_ids={"benchmark"},
        prior_predictions={"frozen": {"predicted_eol_cycle": 2.0, "predicted_rul_cycles": 0.0}},
        created_at="2026-09-02T00:00:00+00:00",
    )

    assert [(row["battery_id"], row["predicted_rul_cycles"], row["predicted_eol_cycle"]) for row in rows] == [
        ("serve", 0.0, 1.0), ("frozen", 0.0, 2.0),
    ]
    assert all(row["selection_revision"] == 2 for row in rows)
