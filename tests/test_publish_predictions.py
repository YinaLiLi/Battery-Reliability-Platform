import pyarrow as pa
import pyarrow.parquet as pq

from src.publish_predictions import constrain_prediction_table, publish


def test_publish_constraint_preserves_raw_prediction_and_freezes_eol():
    source = pa.Table.from_pylist([
        {"model_version": "model", "dataset": "MATR", "battery_id": "B-1", "cycle_index": 2, "predicted_rul_cycles": -2.0},
        {"model_version": "model", "dataset": "MATR", "battery_id": "B-1", "cycle_index": 3, "predicted_rul_cycles": 3.0},
    ])

    constrained = constrain_prediction_table(source).to_pylist()

    assert [(row["predicted_rul_cycles"], row["predicted_eol_cycle"], row["raw_predicted_rul_cycles"]) for row in constrained] == [(0.0, 2.0, -2.0), (0.0, 2.0, 3.0)]


def test_publish_keeps_generation_outputs_in_the_supplied_artifact_directory(tmp_path):
    pq.write_table(pa.Table.from_pylist([{"model_version": "generation", "dataset": "MATR", "battery_id": "B-1", "cycle_index": 1, "predicted_rul_cycles": 2.0}]), tmp_path / "candidate_predictions.parquet")
    pq.write_table(pa.Table.from_pylist([{"model_version": "generation"}]), tmp_path / "candidate_model_evaluation.parquet")

    assert publish(tmp_path) == ("generation", 1)
    assert (tmp_path / "published_predictions.parquet").exists()
    assert (tmp_path / "published_model_evaluation.parquet").exists()
