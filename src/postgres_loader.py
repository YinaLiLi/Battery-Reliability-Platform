"""Load small finalized MATR serving snapshots into PostgreSQL; Parquet remains canonical."""
import argparse
import os
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

DEFAULT_MASTER = "spark://spark-master:7077"
DEFAULT_JDBC_URL = "jdbc:postgresql://postgres:5432/battery_reliability"
ROOT = Path("data/processed/matr")
DATASETS = {
    "battery_cycle_health": {"path": ROOT / "degradation_features", "table": "analytics.battery_cycle_health", "keys": ("dataset", "battery_id", "cycle_index"), "snapshot": True},
    "battery_replay_windows": {"path": ROOT / "replay_cycle_health", "table": "analytics.battery_replay_windows", "keys": ("battery_id", "cycle_index"), "snapshot": True},
    "battery_predictions": {"path": ROOT / "published_predictions.parquet", "table": "analytics.battery_predictions", "keys": ("model_version", "dataset", "battery_id", "cycle_index"), "snapshot": False},
    "model_evaluations": {"path": ROOT / "published_model_evaluation.parquet", "table": "analytics.model_evaluations", "keys": ("model_version",), "snapshot": False},
}

def build_spark_session(master=DEFAULT_MASTER):
    return SparkSession.builder.master(master).appName("matr-postgres-loader").config("spark.sql.session.timeZone", "UTC").config("spark.cores.max", 2).config("spark.executor.cores", 1).config("spark.dynamicAllocation.enabled", "false").getOrCreate()

def prepare_battery_cycle_health(frame):
    return frame.select("dataset", "battery_id", "cycle_index", "soh", "rul_cycles", "discharge_capacity_in_Ah", "internal_resistance_in_ohm", "temperature_max_in_C", "charge_time_in_s", "capacity_slope_10", "coulombic_efficiency")

def prepare_replay_windows(frame):
    return frame.select("battery_id", "cycle_index", "event_count", "average_voltage_in_V", "maximum_temperature_in_C", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "internal_resistance_in_ohm")

def prepare_predictions(frame):
    return frame.select("model_version", "dataset", "battery_id", "cycle_index", "predicted_rul_cycles", F.to_timestamp("prediction_created_at").alias("prediction_created_at"), "split")

def prepare_evaluations(frame):
    return frame.select("model_version", "model_name", "dataset", "status", F.to_timestamp("evaluated_at").alias("evaluated_at"), F.col("metrics_json").alias("metrics"))

PREPARERS = {"battery_cycle_health": prepare_battery_cycle_health, "battery_replay_windows": prepare_replay_windows, "battery_predictions": prepare_predictions, "model_evaluations": prepare_evaluations}

def validate_snapshot(frame, dataset):
    keys = DATASETS[dataset]["keys"]
    if frame.where(F.expr(" OR ".join(f"{key} IS NULL" for key in keys))).limit(1).count() or frame.groupBy(*keys).count().where("count > 1").limit(1).count():
        raise ValueError(f"{dataset} has null or duplicate natural keys")
    if dataset == "battery_cycle_health" and frame.where((F.col("cycle_index") <= 0) | (F.col("soh") < 0)).limit(1).count(): raise ValueError("battery_cycle_health has invalid values")
    if dataset == "battery_replay_windows" and frame.where(F.col("event_count") <= 0).limit(1).count(): raise ValueError("battery_replay_windows has invalid values")

def _execute(spark, jdbc_url, user, password, sql):
    props = spark.sparkContext._jvm.java.util.Properties(); props.setProperty("user", user); props.setProperty("password", password)
    connection = spark.sparkContext._jvm.org.apache.spark.util.Utils.getContextOrSparkClassLoader().loadClass("org.postgresql.Driver").newInstance().connect(jdbc_url, props)
    try:
        statement = connection.createStatement()
        try: statement.executeUpdate(sql)
        finally: statement.close()
    finally: connection.close()

def jdbc_properties(user, password): return {"user": user, "password": password, "driver": "org.postgresql.Driver", "stringtype": "unspecified"}

def assert_group_totals_match(source, target, group_columns, value_column, dataset):
    left = source.groupBy(*group_columns).agg(F.sum(value_column).alias("source_total")); right = target.groupBy(*group_columns).agg(F.sum(value_column).alias("target_total"))
    if left.join(right, list(group_columns), "full").where(F.coalesce(F.col("source_total"), F.lit(0)) != F.coalesce(F.col("target_total"), F.lit(0))).limit(1).count(): raise RuntimeError(f"{dataset} group totals differ after PostgreSQL load")

def load_snapshot(spark, dataset, jdbc_url, user, password, source_path=None):
    config = DATASETS[dataset]; frame = PREPARERS[dataset](spark.read.parquet(str(source_path or config["path"])))
    validate_snapshot(frame, dataset); source_count = frame.count(); properties = jdbc_properties(user, password)
    if config["snapshot"]: _execute(spark, jdbc_url, user, password, f"TRUNCATE TABLE {config['table']}")
    else:
        versions = [row.model_version for row in frame.select("model_version").distinct().collect()]
        if len(versions) != 1: raise ValueError(f"{dataset} must contain exactly one model version")
        _execute(spark, jdbc_url, user, password, f"DELETE FROM {config['table']} WHERE model_version = '{versions[0]}'")
    frame.write.jdbc(jdbc_url, config["table"], mode="append", properties=properties)
    target = spark.read.jdbc(jdbc_url, config["table"], properties=properties)
    if not config["snapshot"]: target = target.where(F.col("model_version") == versions[0])
    if target.count() != source_count or target.groupBy(*config["keys"]).count().where("count > 1").limit(1).count(): raise RuntimeError(f"{dataset} did not match PostgreSQL after load")
    return source_count

def parse_args():
    parser = argparse.ArgumentParser(description="Load MATR serving snapshots into PostgreSQL.")
    parser.add_argument("--dataset", choices=DATASETS, required=True); parser.add_argument("--master", default=DEFAULT_MASTER); parser.add_argument("--jdbc-url", default=os.environ.get("POSTGRES_JDBC_URL", DEFAULT_JDBC_URL)); parser.add_argument("--jdbc-user", default=os.environ.get("POSTGRES_USER", "battery_reliability")); parser.add_argument("--jdbc-password", default=os.environ.get("POSTGRES_PASSWORD"), required=os.environ.get("POSTGRES_PASSWORD") is None); parser.add_argument("--source-path", type=Path)
    return parser.parse_args()

def main():
    args = parse_args(); spark = build_spark_session(args.master); spark.sparkContext.setLogLevel("WARN")
    try: print(f"Loaded {load_snapshot(spark, args.dataset, args.jdbc_url, args.jdbc_user, args.jdbc_password, args.source_path)} {args.dataset} row(s).")
    finally: spark.stop()

if __name__ == "__main__": main()
