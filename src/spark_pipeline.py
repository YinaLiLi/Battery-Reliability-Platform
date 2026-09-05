"""Build leakage-safe cycle-level MATR degradation features with PySpark."""
import argparse
import json
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

try:
    from .feature_contract import (
        render_spark_cycle_aggregate,
        spark_causal_features,
        spark_cycle_aggregate_expressions,
    )
    from .stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
except ImportError:
    from feature_contract import render_spark_cycle_aggregate, spark_causal_features, spark_cycle_aggregate_expressions
    from stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest


SHUFFLE_PARTITIONS = 3
_CYCLE_INPUTS = {
    "internal_resistance_in_ohm", "temperature_min_in_C", "temperature_max_in_C", "charge_time_in_s",
    "voltage_min_in_V", "voltage_max_in_V", "voltage_mean_in_V", "current_mean_in_A", "current_abs_max_in_A",
    "charge_capacity_in_Ah", "discharge_capacity_in_Ah",
}


def build_spark_session(master=None):
    master = master or os.environ.get("SPARK_MASTER", "local[*]")
    try:
        from .spark_environment import configure_local_python
    except ImportError:
        from spark_environment import configure_local_python
    configure_local_python(master)
    return SparkSession.builder.master(master).appName("matr-degradation-features").config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS).getOrCreate()


def _measurement_aggregates(measurements):
    """Use the same telemetry reducer as the Streaming state builder."""
    return render_spark_cycle_aggregate(
        measurements.groupBy("battery_id", "cycle_index").agg(*spark_cycle_aggregate_expressions(F, measurements.columns)), F
    )


def build_features(cycles, provenance, measurements=None):
    """Features use only current/prior rows under the shared causal contract."""
    frame = cycles.join(provenance.select("battery_id", "lineage_group_id", "batch_id", "charge_policy"), "battery_id")
    if measurements is not None:
        aggregates = _measurement_aggregates(measurements)
        for column in _CYCLE_INPUTS:
            aggregates = aggregates.withColumnRenamed(column, f"_stream_{column}")
        frame = frame.join(aggregates, ["battery_id", "cycle_index"], "left")
        for column in _CYCLE_INPUTS:
            source = F.col(f"_stream_{column}")
            frame = frame.withColumn(column, F.coalesce(source, F.col(column)) if column in cycles.columns else source)
    return spark_causal_features(frame, F, Window)


def _state_boundary(spark, manifest_path, boundary_path):
    manifest = json.loads(Path(manifest_path).read_text())
    boundary = json.loads(Path(boundary_path).read_text())
    validate_stream_state_manifest(
        manifest, boundary, expected_canonical_fingerprint=manifest["canonical_fingerprint"],
        expected_arrival_manifest_fingerprint=manifest["arrival_manifest_fingerprint"],
        expected_feature_contract_version=manifest["feature_contract_version"],
    )
    return manifest, spark.createDataFrame(validate_finalized_cycle_boundary(boundary)["finalized_cycle_keys"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=Path, default=Path("data/processed/matr/cycle_summary"))
    parser.add_argument("--measurements", type=Path, default=Path("data/processed/matr/cycle_measurements"))
    parser.add_argument("--provenance", type=Path, default=Path("data/processed/matr/matr_provenance.parquet"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--finalized-cycle-boundary", type=Path)
    parser.add_argument("--arrival-manifest", type=Path, default=Path("data/processed/matr/arrival_manifest.parquet"))
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if bool(args.state_manifest) != bool(args.finalized_cycle_boundary):
        raise SystemExit("--state-manifest and --finalized-cycle-boundary must be supplied together")
    spark = build_spark_session(args.master)
    try:
        cycles = spark.read.parquet(str(args.cycles))
        measurements = spark.read.parquet(str(args.measurements))
        provenance = spark.read.parquet(str(args.provenance))
        output = args.output or Path("data/processed/matr/degradation_features")
        if args.state_manifest:
            manifest, boundary = _state_boundary(spark, args.state_manifest, args.finalized_cycle_boundary)
            cycles = cycles.join(boundary, ["dataset", "battery_id", "cycle_index"])
            measurements = measurements.join(boundary, ["dataset", "battery_id", "cycle_index"])
            if args.train_only:
                splits = spark.read.parquet(str(args.arrival_manifest)).select("battery_id", "split")
                train = splits.where(F.col("split") == "train").select("battery_id")
                cycles, measurements = cycles.join(train, "battery_id"), measurements.join(train, "battery_id")
            output = args.output or Path("data/processed/matr/historical_features") / manifest["state_id"]
        build_features(cycles, provenance, measurements).write.mode("overwrite").parquet(str(output))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
