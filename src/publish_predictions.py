"""Promote the freshly evaluated candidate files to the current Parquet serving snapshot."""
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path("data/processed/matr")

def main():
    predictions = pq.read_table(ROOT / "candidate_predictions.parquet")
    evaluation = pq.read_table(ROOT / "candidate_model_evaluation.parquet")
    if predictions.num_rows == 0 or evaluation.num_rows != 1:
        raise ValueError("candidate prediction/evaluation artifacts are incomplete")
    pq.write_table(predictions, ROOT / "published_predictions.parquet")
    pq.write_table(evaluation, ROOT / "published_model_evaluation.parquet")
    print(f"Published candidate {evaluation.column('model_version')[0].as_py()} with {predictions.num_rows} predictions.")

if __name__ == "__main__": main()
