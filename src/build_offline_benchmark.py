"""Materialize the immutable complete validation/test RUL benchmark once."""

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from offline_benchmark import build_fixed_benchmark, select_benchmark_rows
from train_matr_models import FEATURE_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/matr/degradation_features"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/matr/arrival_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/matr/fixed_offline_benchmark/v1"))
    args = parser.parse_args()
    rows = ds.dataset(args.features, format="parquet").to_table().to_pylist()
    manifest = pq.read_table(args.manifest).to_pylist()
    benchmark = build_fixed_benchmark(manifest, rows, feature_contract_version=FEATURE_VERSION)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "benchmark.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != benchmark:
        raise ValueError("fixed offline benchmark already exists with different content")
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(select_benchmark_rows(rows, benchmark, "validation") + select_benchmark_rows(rows, benchmark, "test")), args.output / "held_out_features.parquet")
    metadata_path.write_text(json.dumps(benchmark, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
