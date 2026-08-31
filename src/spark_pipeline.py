"""Build labeled EV telemetry features with PySpark on the Docker Spark cluster."""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


DEFAULT_TELEMETRY_PATH = Path("data/processed/synthetic_fleet_telemetry.parquet")
DEFAULT_LABELS_PATH = Path("data/processed/synthetic_fleet_labels.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/spark_batch_features")
DEFAULT_MASTER = "spark://spark-master:7077"
SHUFFLE_PARTITIONS = 3


def build_spark_session(master=DEFAULT_MASTER):
    """Create the small cluster session used for plan inspection and exercises."""
    return (
        SparkSession.builder.master(master)
        .appName("ev-fleet-batch-foundations")
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.sql.autoBroadcastJoinThreshold", -1)
        .config("spark.cores.max", 2)
        .config("spark.executor.cores", 1)
        .config("spark.dynamicAllocation.enabled", "false")
        .getOrCreate()
    )


def clean_telemetry(telemetry):
    """Normalize timestamp and add the first telemetry-only feature."""
    return telemetry.withColumn("timestamp", F.to_timestamp("timestamp")).withColumn(
        "module_temp_spread", F.col("module_temp_max") - F.col("module_temp_min")
    )


def vehicle_dimension(telemetry):
    """Derive one stable vehicle record for the broadcast join."""
    return telemetry.groupBy("vehicle_id").agg(
        F.first("battery_type", ignorenulls=True).alias("vehicle_battery_type"),
        F.first("region", ignorenulls=True).alias("vehicle_region"),
    )


def build_features(telemetry, labels):
    """Return labeled telemetry with vehicle-window and broadcast-dimension features."""
    cleaned = clean_telemetry(telemetry)
    labels = labels.select("event_id", "failure_within_30_operating_days")
    history = Window.partitionBy("vehicle_id").orderBy("timestamp").rowsBetween(-23, 0)
    prior_event = Window.partitionBy("vehicle_id").orderBy("timestamp")
    featured = (
        cleaned.withColumn("previous_pack_voltage", F.lag("pack_voltage").over(prior_event))
        .withColumn("rolling_module_temp_max", F.max("module_temp_max").over(history))
        .join(labels, "event_id", "inner")
    )
    return featured.join(F.broadcast(vehicle_dimension(cleaned)), "vehicle_id", "inner")


def partition_counts(frame, key):
    """Return row counts per physical partition after repartitioning on key."""
    return (
        frame.repartition(SHUFFLE_PARTITIONS, key)
        .select(F.spark_partition_id().alias("partition"))
        .groupBy("partition")
        .count()
        .orderBy("partition")
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build labeled EV telemetry features with PySpark.")
    parser.add_argument("--telemetry-path", type=Path, default=DEFAULT_TELEMETRY_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = build_spark_session(args.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        telemetry = clean_telemetry(spark.read.parquet(str(args.telemetry_path))).cache()
        telemetry_count = telemetry.count()  # action: materializes the cache for both downstream branches
        labels = spark.read.parquet(str(args.labels_path))
        featured = build_features(telemetry, labels)

        print(f"Telemetry rows: {telemetry_count}; partitions: {telemetry.rdd.getNumPartitions()}")
        print(f"Cached telemetry: {telemetry.is_cached}")
        print("\nShuffle label join and broadcast vehicle-dimension plan:")
        featured.explain("formatted")
        print("\nRows per partition after repartitioning by vehicle_id:")
        partition_counts(telemetry, "vehicle_id").show()
        print("Rows per partition after repartitioning by is_charging (intentional skew example):")
        partition_counts(telemetry, "is_charging").show()

        output = featured.repartition(SHUFFLE_PARTITIONS, "region")
        output.write.mode("overwrite").partitionBy("region").parquet(str(args.output_path))
        print(f"Labeled feature rows: {featured.count()}; wrote {args.output_path}")
    finally:
        if "telemetry" in locals():
            telemetry.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
