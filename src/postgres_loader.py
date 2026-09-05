"""Load small finalized MATR serving snapshots into PostgreSQL; Parquet remains canonical."""
import argparse
import os
import re
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

DEFAULT_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
DEFAULT_JDBC_URL = "jdbc:postgresql://localhost:5432/battery_reliability"
ROOT = Path("data/processed/matr")
DATASETS = {
    "battery_cycle_health": {"path": ROOT / "degradation_features", "table": "analytics.battery_cycle_health", "keys": ("dataset", "battery_id", "cycle_index"), "snapshot": True},
    "battery_replay_windows": {"path": ROOT / "replay_cycle_health", "table": "analytics.battery_replay_windows", "keys": ("battery_id", "cycle_index"), "snapshot": True},
    "battery_predictions": {"path": ROOT / "published_predictions.parquet", "table": "analytics.battery_predictions", "keys": ("model_version", "dataset", "battery_id", "cycle_index"), "snapshot": False},
    "model_evaluations": {"path": ROOT / "published_model_evaluation.parquet", "table": "analytics.model_evaluations", "keys": ("model_version",), "snapshot": False},
    "battery_survival_predictions": {"path": ROOT / "published_survival_predictions.parquet", "table": "analytics.battery_survival_predictions", "keys": ("model_version", "dataset", "battery_id", "cycle_index", "horizon_cycles"), "snapshot": False},
    "survival_model_evaluations": {"path": ROOT / "published_survival_model_evaluation.parquet", "table": "analytics.survival_model_evaluations", "keys": ("model_version",), "snapshot": False},
}

def build_spark_session(master=DEFAULT_MASTER):
    try:
        from .spark_environment import configure_local_python
    except ImportError:
        from spark_environment import configure_local_python
    configure_local_python(master)
    return SparkSession.builder.master(master).appName("matr-postgres-loader").config("spark.sql.session.timeZone", "UTC").config("spark.cores.max", 2).config("spark.executor.cores", 1).config("spark.dynamicAllocation.enabled", "false").getOrCreate()

def prepare_battery_cycle_health(frame):
    return frame.select("dataset", "battery_id", "cycle_index", "soh", "rul_cycles", "discharge_capacity_in_Ah", "internal_resistance_in_ohm", "temperature_max_in_C", "charge_time_in_s", "capacity_slope_10", "coulombic_efficiency")

def prepare_replay_windows(frame):
    return frame.select("battery_id", "cycle_index", "event_count", "average_voltage_in_V", "maximum_temperature_in_C", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "internal_resistance_in_ohm")

def prepare_predictions(frame):
    raw = F.col("raw_predicted_rul_cycles") if "raw_predicted_rul_cycles" in frame.columns else F.col("predicted_rul_cycles")
    eol = F.col("predicted_eol_cycle") if "predicted_eol_cycle" in frame.columns else F.col("cycle_index") + F.col("predicted_rul_cycles")
    return frame.select("model_version", "dataset", "battery_id", "cycle_index", raw.alias("raw_predicted_rul_cycles"), "predicted_rul_cycles", eol.alias("predicted_eol_cycle"), F.to_timestamp("prediction_created_at").alias("prediction_created_at"), "split")

def prepare_evaluations(frame):
    metadata = F.col("training_metadata_json") if "training_metadata_json" in frame.columns else F.lit("{}")
    fingerprint = F.col("model_fingerprint") if "model_fingerprint" in frame.columns else F.lit(None).cast("string")
    generation = F.col("generation").cast("string") if "generation" in frame.columns else F.lit(None).cast("string")
    return frame.select("model_version", "model_name", "dataset", "status", F.to_timestamp("evaluated_at").alias("evaluated_at"), F.col("metrics_json").alias("metrics"), metadata.alias("training_metadata"), fingerprint.alias("model_fingerprint"), generation.alias("generation"))

def prepare_survival_predictions(frame):
    return frame.select("model_version", "dataset", "battery_id", "cycle_index", "horizon_cycles", "survival_probability", F.to_timestamp("prediction_created_at").alias("prediction_created_at"), "split")

PREPARERS = {"battery_cycle_health": prepare_battery_cycle_health, "battery_replay_windows": prepare_replay_windows, "battery_predictions": prepare_predictions, "model_evaluations": prepare_evaluations, "battery_survival_predictions": prepare_survival_predictions, "survival_model_evaluations": prepare_evaluations}

def build_current_prediction_upsert_sql(staging_table):
    """Build the idempotent merge for finalized streaming predictions."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", staging_table):
        raise ValueError("staging_table must be a safe PostgreSQL identifier")
    columns = "dataset, battery_id, model_version, model_fingerprint, state_id, replay_sequence, cycle_index, raw_predicted_rul_cycles, predicted_rul_cycles, predicted_eol_cycle, inference_created_at, selection_revision"
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in columns.split(", ") if column not in {"dataset", "battery_id"})
    target = "analytics.battery_current_predictions"
    return f"""INSERT INTO {target} ({columns})
SELECT {columns} FROM {staging_table}
ON CONFLICT (dataset, battery_id) DO UPDATE SET {updates}
WHERE EXCLUDED.replay_sequence > {target}.replay_sequence
   OR (EXCLUDED.replay_sequence = {target}.replay_sequence
       AND EXCLUDED.state_id = {target}.state_id
       AND EXCLUDED.selection_revision > {target}.selection_revision)"""

def build_current_survival_prediction_upsert_sql(staging_table):
    """Build the idempotent merge for current survival serving rows."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", staging_table):
        raise ValueError("staging_table must be a safe PostgreSQL identifier")
    columns = "dataset, battery_id, cycle_index, horizon_cycles, survival_probability, model_version, model_fingerprint, state_id, replay_sequence, feature_contract_version, selection_revision, inference_created_at"
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in columns.split(", ") if column not in {"dataset", "battery_id", "horizon_cycles"})
    target = "analytics.battery_current_survival_predictions"
    return f"""INSERT INTO {target} ({columns})
SELECT {columns} FROM {staging_table}
ON CONFLICT (dataset, battery_id, horizon_cycles) DO UPDATE SET {updates}
WHERE EXCLUDED.replay_sequence > {target}.replay_sequence
   OR (EXCLUDED.replay_sequence = {target}.replay_sequence
       AND EXCLUDED.state_id = {target}.state_id
       AND EXCLUDED.selection_revision > {target}.selection_revision)"""

def build_current_stream_state_upsert_sql(staging_table):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", staging_table):
        raise ValueError("staging_table must be a safe PostgreSQL identifier")
    return f"""INSERT INTO analytics.current_stream_states (dataset, state_id, feature_contract_version, published_at)
SELECT dataset, state_id, feature_contract_version, published_at FROM {staging_table}
ON CONFLICT (dataset) DO UPDATE SET state_id = EXCLUDED.state_id,
    feature_contract_version = EXCLUDED.feature_contract_version, published_at = EXCLUDED.published_at"""

def build_serving_status_upsert_sql(staging_table):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", staging_table):
        raise ValueError("staging_table must be a safe PostgreSQL identifier")
    columns = "dataset, state_id, consumer, selection_revision, model_version, model_fingerprint, status, rows_written, error_message, updated_at"
    return f"""INSERT INTO analytics.stream_serving_status ({columns})
SELECT {columns} FROM {staging_table}
ON CONFLICT (dataset, state_id, consumer, selection_revision) DO UPDATE SET
    model_version = EXCLUDED.model_version, model_fingerprint = EXCLUDED.model_fingerprint,
    status = EXCLUDED.status, rows_written = EXCLUDED.rows_written,
    error_message = EXCLUDED.error_message, updated_at = EXCLUDED.updated_at"""

def validate_snapshot(frame, dataset):
    keys = DATASETS[dataset]["keys"]
    if frame.where(F.expr(" OR ".join(f"{key} IS NULL" for key in keys))).limit(1).count() or frame.groupBy(*keys).count().where("count > 1").limit(1).count():
        raise ValueError(f"{dataset} has null or duplicate natural keys")
    if dataset == "battery_cycle_health" and frame.where((F.col("cycle_index") <= 0) | (F.col("soh") < 0)).limit(1).count(): raise ValueError("battery_cycle_health has invalid values")
    if dataset == "battery_predictions":
        if frame.where(F.col("predicted_rul_cycles") < 0).limit(1).count(): raise ValueError("battery_predictions has negative served RUL")
        from pyspark.sql.window import Window
        trajectory = Window.partitionBy("model_version", "dataset", "battery_id").orderBy("cycle_index").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        first_eol = F.min(F.when(F.col("predicted_rul_cycles") == 0, F.col("cycle_index"))).over(trajectory)
        checked = frame.withColumn("first_eol", first_eol)
        if checked.where(((F.col("first_eol").isNotNull()) & ((F.col("predicted_rul_cycles") != 0) | (F.col("predicted_eol_cycle") != F.col("first_eol")))) | ((F.col("first_eol").isNull()) & (F.col("predicted_eol_cycle") != F.col("cycle_index") + F.col("predicted_rul_cycles")))).limit(1).count(): raise ValueError("battery_predictions violates irreversible predicted EOL")
    if dataset == "battery_survival_predictions":
        if frame.where((F.col("horizon_cycles") < 0) | (F.col("horizon_cycles") > 200) | (F.col("survival_probability") < 0) | (F.col("survival_probability") > 1) | F.isnan("survival_probability")).limit(1).count(): raise ValueError("battery_survival_predictions has invalid probabilities")
        from pyspark.sql.window import Window
        curve = Window.partitionBy("model_version", "dataset", "battery_id", "cycle_index").orderBy("horizon_cycles")
        checked = frame.withColumn("prior_probability", F.lag("survival_probability").over(curve))
        if checked.where((F.col("survival_probability") > F.col("prior_probability")) | ((F.col("horizon_cycles") == 0) & (F.col("survival_probability") != 1))).limit(1).count(): raise ValueError("battery_survival_predictions must be non-increasing from one")
        required = frame.where(F.col("horizon_cycles").isin(50, 100, 200)).groupBy("model_version", "dataset", "battery_id", "cycle_index").count()
        if required.where(F.col("count") != 3).limit(1).count() or required.count() != frame.select("model_version", "dataset", "battery_id", "cycle_index").distinct().count(): raise ValueError("battery_survival_predictions missing required horizons")
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
