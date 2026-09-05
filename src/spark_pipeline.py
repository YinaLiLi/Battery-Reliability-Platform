"""Build leakage-safe cycle-level MATR degradation features with PySpark."""
import argparse
import json
import os
from pathlib import Path
import shutil

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

try:
    from .feature_contract import (
        SHARED_FEATURE_COLUMNS,
        render_spark_cycle_aggregate,
        spark_causal_features,
        spark_cycle_aggregate_expressions,
    )
    from .stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    from .shared_features import append_shared_feature_rows, finalize_shared_features, stage_shared_features, validate_shared_features
except ImportError:
    from feature_contract import SHARED_FEATURE_COLUMNS, render_spark_cycle_aggregate, spark_causal_features, spark_cycle_aggregate_expressions
    from stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    from shared_features import append_shared_feature_rows, finalize_shared_features, stage_shared_features, validate_shared_features


SHUFFLE_PARTITIONS = 3
KEY_COLUMNS = ("dataset", "battery_id", "cycle_index")
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
    validate_finalized_cycle_boundary(boundary)
    rows = boundary.get("finalized_cycle_keys") or boundary.get("finalized_cycle_ranges")
    return manifest, spark.createDataFrame(rows), "finalized_cycle_keys" in boundary


def materialize_shared_features(frame, output, *, state_manifest_path, generation):
    """Create one immutable state-bound feature dataset, or validate its retry."""
    output = Path(output)
    if output.exists():
        return validate_shared_features(
            output, state_manifest_path=state_manifest_path, generation=generation
        )
    staged = stage_shared_features(output)
    try:
        frame.write.mode("errorifexists").parquet(str(staged))
        return finalize_shared_features(
            staged, output, state_manifest_path=state_manifest_path, generation=generation
        )
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def append_feature_outlet(frame, output, *, generation, feature_contract_version, canonical_source_fingerprint):
    """Append only normalized derived columns from a completed-cycle frame."""
    columns = [*KEY_COLUMNS, *(column for column in SHARED_FEATURE_COLUMNS if column in frame.columns)]
    if len(columns) == len(KEY_COLUMNS):
        raise ValueError("completed cycles contain no shared derived features")
    rows = [row.asDict(recursive=True) for row in frame.select(*columns).collect()]
    return append_shared_feature_rows(
        output, rows, generation=generation, feature_contract_version=feature_contract_version,
        canonical_source_fingerprint=canonical_source_fingerprint,
    )


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
    parser.add_argument("--generation")
    args = parser.parse_args()
    if bool(args.state_manifest) != bool(args.finalized_cycle_boundary):
        raise SystemExit("--state-manifest and --finalized-cycle-boundary must be supplied together")
    spark = build_spark_session(args.master)
    try:
        cycles = spark.read.parquet(str(args.cycles))
        measurements = spark.read.parquet(str(args.measurements))
        provenance = spark.read.parquet(str(args.provenance))
        output = args.output or Path("data/processed/matr/shared_feature_outlet")
        if args.state_manifest:
            manifest, boundary, explicit_keys = _state_boundary(spark, args.state_manifest, args.finalized_cycle_boundary)
            if explicit_keys:
                cycles = cycles.join(boundary, ["dataset", "battery_id", "cycle_index"])
                measurements = measurements.join(boundary, ["dataset", "battery_id", "cycle_index"])
            else:
                ranges = boundary.select("dataset", "battery_id", "max_finalized_cycle_index")
                condition = ["dataset", "battery_id"]
                cycles = cycles.join(ranges, condition).where(F.col("cycle_index") <= F.col("max_finalized_cycle_index")).drop("max_finalized_cycle_index")
                measurements = measurements.join(ranges, condition).where(F.col("cycle_index") <= F.col("max_finalized_cycle_index")).drop("max_finalized_cycle_index")
            if args.train_only:
                splits = spark.read.parquet(str(args.arrival_manifest)).select("battery_id", "split")
                train = splits.where(F.col("split") == "train").select("battery_id")
                cycles, measurements = cycles.join(train, "battery_id"), measurements.join(train, "battery_id")
        features = build_features(cycles, provenance, measurements)
        if args.state_manifest:
            generation = args.generation or manifest.get("generation") or manifest.get("cutoff_metadata", {}).get("generation")
            if not generation:
                raise ValueError("--generation is required for shared feature outlet append")
            append_feature_outlet(
                features, output, generation=generation,
                feature_contract_version=manifest["feature_contract_version"],
                canonical_source_fingerprint=manifest["canonical_fingerprint"],
            )
        else:
            raise ValueError("outlet bootstrap requires --state-manifest, --finalized-cycle-boundary, and --generation")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
