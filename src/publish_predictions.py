"""Publish one candidate generation without moving its immutable artifacts."""
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from .rul_predictions import constrain_prediction_trajectory
except ImportError:
    from rul_predictions import constrain_prediction_trajectory

ROOT = Path("data/processed/matr")


def constrain_prediction_table(predictions):
    trajectories = {}
    for row in predictions.to_pylist():
        key = (row["model_version"], row["dataset"], row["battery_id"])
        trajectories.setdefault(key, []).append(row)
    return pa.Table.from_pylist([
        row for trajectory in trajectories.values() for row in constrain_prediction_trajectory(trajectory)
    ])


def publish(artifact_dir=ROOT):
    artifact_dir = Path(artifact_dir)
    predictions = constrain_prediction_table(pq.read_table(artifact_dir / "candidate_predictions.parquet"))
    evaluation = pq.read_table(artifact_dir / "candidate_model_evaluation.parquet")
    if predictions.num_rows == 0 or evaluation.num_rows != 1:
        raise ValueError("candidate prediction/evaluation artifacts are incomplete")
    pq.write_table(predictions, artifact_dir / "published_predictions.parquet")
    pq.write_table(evaluation, artifact_dir / "published_model_evaluation.parquet")
    return evaluation.column("model_version")[0].as_py(), predictions.num_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    model_version, count = publish(args.artifact_dir)
    print(f"Published candidate {model_version} with {count} predictions.")

if __name__ == "__main__": main()
