"""Publish an immutable survival candidate without applying RUL constraints."""
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .survival_models import validate_prediction_rows
except ImportError:
    from survival_models import validate_prediction_rows


ROOT = Path("data/processed/matr")


def publish(artifact_dir=ROOT):
    artifact_dir = Path(artifact_dir)
    predictions = pq.read_table(artifact_dir / "candidate_survival_predictions.parquet")
    evaluation = pq.read_table(artifact_dir / "candidate_survival_model_evaluation.parquet")
    rows = predictions.to_pylist()
    groups = {}
    for row in rows:
        groups.setdefault((row["model_version"], row["dataset"], row["battery_id"], row["cycle_index"]), []).append(row)
    for curve in groups.values():
        validate_prediction_rows(curve)
    if not rows or evaluation.num_rows != 1:
        raise ValueError("candidate survival artifacts are incomplete")
    pq.write_table(predictions, artifact_dir / "published_survival_predictions.parquet")
    pq.write_table(evaluation, artifact_dir / "published_survival_model_evaluation.parquet")
    return evaluation.column("model_version")[0].as_py(), len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    version, count = publish(args.artifact_dir)
    print(f"Published survival candidate {version} with {count} predictions.")
