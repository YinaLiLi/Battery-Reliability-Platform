"""Load finalized Spark Parquet snapshots into PostgreSQL analytics tables."""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


DEFAULT_MASTER = "spark://spark-master:7077"
DEFAULT_JDBC_URL = "jdbc:postgresql://postgres:5432/ev_fleet"
DEFAULT_FEATURE_PATH = Path("data/processed/spark_batch_features")
DEFAULT_WINDOW_PATH = Path("data/processed/spark_streaming_windows")

DATASETS = {
    "vehicle_features": {
        "path": DEFAULT_FEATURE_PATH,
        "table": "analytics.vehicle_features",
        "keys": ("event_id",),
    },
    "vehicle_window_metrics": {
        "path": DEFAULT_WINDOW_PATH,
        "table": "analytics.vehicle_window_metrics",
        "keys": ("window_start", "window_end", "vehicle_id"),
    },
}


def build_spark_session(master=DEFAULT_MASTER):
    """Create the Spark session used for snapshot loading."""
    return (
        SparkSession.builder.master(master)
        .appName("ev-fleet-postgres-loader")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.cores.max", 2)
        .config("spark.executor.cores", 1)
        .config("spark.dynamicAllocation.enabled", "false")
        .getOrCreate()
    )


def prepare_vehicle_features(frame):
    """Rename the batch timestamp to the persistent analytics column name."""
    return frame.withColumnRenamed("timestamp", "event_time").withColumn(
        "failure_within_30_operating_days", F.col("failure_within_30_operating_days").cast("short")
    )


def prepare_window_metrics(frame):
    """Flatten Spark's window struct into the PostgreSQL primary-key columns."""
    return frame.select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        "vehicle_id",
        "event_count",
        "average_pack_voltage",
        "maximum_module_temperature",
    )


def validate_snapshot(frame, dataset):
    """Fail before truncation if a finalized snapshot violates table invariants."""
    keys = DATASETS[dataset]["keys"]
    if frame.where(F.expr(" OR ".join(f"{key} IS NULL" for key in keys))).limit(1).count():
        raise ValueError(f"{dataset} contains a null primary-key value")
    if frame.groupBy(*keys).count().where("count > 1").limit(1).count():
        raise ValueError(f"{dataset} contains duplicate primary-key values")
    if dataset == "vehicle_features":
        invalid = frame.where(
            ~F.col("failure_within_30_operating_days").isin(0, 1)
            | (F.col("module_temp_min") > F.col("module_temp_max"))
        )
    else:
        invalid = frame.where(
            (F.col("event_count") <= 0) | (F.col("window_start") >= F.col("window_end"))
        )
    if invalid.limit(1).count():
        raise ValueError(f"{dataset} contains invalid analytics values")


def truncate_target(spark, jdbc_url, user, password, table):
    """Explicitly truncate a pre-created table without allowing Spark to recreate it."""
    jvm = spark.sparkContext._jvm
    properties = jvm.java.util.Properties()
    properties.setProperty("user", user)
    properties.setProperty("password", password)
    driver_class = jvm.org.apache.spark.util.Utils.getContextOrSparkClassLoader().loadClass("org.postgresql.Driver")
    connection = driver_class.newInstance().connect(jdbc_url, properties)
    try:
        statement = connection.createStatement()
        try:
            statement.executeUpdate(f"TRUNCATE TABLE {table}")
        finally:
            statement.close()
    finally:
        connection.close()


def jdbc_properties(user, password):
    return {"user": user, "password": password, "driver": "org.postgresql.Driver"}


def assert_group_totals_match(source, target, group_columns, value_column, dataset):
    """Confirm dashboard-level totals agree with the finalized Parquet snapshot."""
    source_totals = source.groupBy(*group_columns).agg(F.sum(value_column).alias("source_total"))
    target_totals = target.groupBy(*group_columns).agg(F.sum(value_column).alias("target_total"))
    mismatch = source_totals.join(target_totals, list(group_columns), "full").where(
        F.coalesce(F.col("source_total"), F.lit(0)) != F.coalesce(F.col("target_total"), F.lit(0))
    )
    if mismatch.limit(1).count():
        raise RuntimeError(f"{dataset} group totals differ after PostgreSQL load")


def load_snapshot(spark, dataset, jdbc_url, user, password, source_path=None):
    """Validate, truncate, append, and verify one complete Parquet snapshot."""
    config = DATASETS[dataset]
    source = source_path or config["path"]
    frame = spark.read.parquet(str(source))
    prepared = prepare_vehicle_features(frame) if dataset == "vehicle_features" else prepare_window_metrics(frame)
    validate_snapshot(prepared, dataset)
    source_count = prepared.count()
    properties = jdbc_properties(user, password)
    truncate_target(spark, jdbc_url, user, password, config["table"])
    prepared.write.jdbc(jdbc_url, config["table"], mode="append", properties=properties)
    target = spark.read.jdbc(jdbc_url, config["table"], properties=properties)
    if target.count() != source_count:
        raise RuntimeError(f"{dataset} row count differs after PostgreSQL load")
    if target.groupBy(*config["keys"]).count().where("count > 1").limit(1).count():
        raise RuntimeError(f"{dataset} contains duplicate primary-key values after PostgreSQL load")
    if dataset == "vehicle_features":
        assert_group_totals_match(
            prepared.withColumn("row_count", F.lit(1)), target.withColumn("row_count", F.lit(1)), ("region",), "row_count", dataset
        )
    else:
        assert_group_totals_match(prepared, target, ("window_start", "window_end"), "event_count", dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Load a finalized EV Fleet Parquet snapshot into PostgreSQL.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--jdbc-url", default=os.environ.get("POSTGRES_JDBC_URL", DEFAULT_JDBC_URL))
    parser.add_argument("--jdbc-user", default=os.environ.get("POSTGRES_USER", "ev_fleet"))
    parser.add_argument("--jdbc-password", default=os.environ.get("POSTGRES_PASSWORD"), required=os.environ.get("POSTGRES_PASSWORD") is None)
    parser.add_argument("--source-path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = build_spark_session(args.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        load_snapshot(spark, args.dataset, args.jdbc_url, args.jdbc_user, args.jdbc_password, args.source_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
