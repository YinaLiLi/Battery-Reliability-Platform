import pyarrow as pa
import pyarrow.parquet as pq

from src.publish_survival_predictions import publish


def test_publish_survival_predictions_copies_one_valid_candidate(tmp_path):
    rows = [
        {"model_version": "survival-1", "dataset": "MATR", "battery_id": "B-1", "cycle_index": 10, "horizon_cycles": horizon, "survival_probability": probability, "prediction_created_at": "2026-09-02T00:00:00+00:00", "split": "test"}
        for horizon, probability in ((0, 1.0), (50, 0.9), (100, 0.8), (200, 0.7))
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "candidate_survival_predictions.parquet")
    pq.write_table(pa.Table.from_pylist([{"model_version": "survival-1"}]), tmp_path / "candidate_survival_model_evaluation.parquet")

    assert publish(tmp_path) == ("survival-1", 4)
    assert (tmp_path / "published_survival_predictions.parquet").exists()
