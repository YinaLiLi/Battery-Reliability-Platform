"""Aggregate Kafka telemetry with PySpark Structured Streaming."""

import argparse
import json
import os
import re
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T
try:
    from .stream_runtime import publish_state_artifacts, publish_latest_manifest
    from .stream_state import build_compact_finalized_cycle_boundary, state_id_for_boundary
    from .train_matr_models import FEATURE_VERSION
    from .feature_contract import RUL_FEATURES, SHARED_FEATURE_COLUMNS, feature_rows, render_spark_cycle_aggregate, spark_cycle_aggregate_expressions
    from .shared_features import append_shared_feature_rows, feature_outlet_key_maxima, generation_for_timestamp, load_current_feature_rows
    from .stream_inference import current_prediction_rows
    from .serving_status import current_stream_state_row, serving_status_row
    from .postgres_loader import build_current_prediction_upsert_sql, build_current_stream_state_upsert_sql, build_serving_status_upsert_sql, _execute, jdbc_properties
except ImportError:
    from stream_runtime import publish_state_artifacts, publish_latest_manifest
    from stream_state import build_compact_finalized_cycle_boundary, state_id_for_boundary
    from train_matr_models import FEATURE_VERSION
    from feature_contract import RUL_FEATURES, SHARED_FEATURE_COLUMNS, feature_rows, render_spark_cycle_aggregate, spark_cycle_aggregate_expressions
    from shared_features import append_shared_feature_rows, feature_outlet_key_maxima, generation_for_timestamp, load_current_feature_rows
    from stream_inference import current_prediction_rows
    from serving_status import current_stream_state_row, serving_status_row
    from postgres_loader import build_current_prediction_upsert_sql, build_current_stream_state_upsert_sql, build_serving_status_upsert_sql, _execute, jdbc_properties


DEFAULT_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
DEFAULT_BOOTSTRAP_SERVER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_OUTPUT_PATH = Path("data/processed/matr/replay_cycle_health")
DEFAULT_CHECKPOINT_PATH = Path("data/processed/matr/replay_checkpoint")
DEFAULT_LIFECYCLE_OUTPUT_PATH = Path("data/processed/matr/replay_lifecycle_state")
DEFAULT_COMPLETED_CYCLES_PATH = Path("data/processed/matr/replay_completed_cycles")
DEFAULT_KAFKA_OFFSETS_PATH = Path("data/processed/matr/replay_kafka_offsets")
DEFAULT_STATE_ROOT = Path("data/processed/matr")
DEFAULT_ARRIVAL_MANIFEST_PATH = Path("data/processed/matr/arrival_manifest.parquet")
DEFAULT_CANONICAL_CYCLES_PATH = Path("data/processed/matr/cycle_summary")
TOPIC = "battery_measurements"
LIFECYCLE_TOPIC = "battery_lifecycle"
SHUFFLE_PARTITIONS = 3
MAX_OFFSETS_PER_TRIGGER = 100
WATERMARK_DELAY = "2 hours"

TELEMETRY_SCHEMA = T.StructType(
    [
        T.StructField("event_id", T.StringType()),
        T.StructField("dataset", T.StringType()),
        T.StructField("battery_id", T.StringType()),
        T.StructField("cycle_index", T.IntegerType()),
        T.StructField("sample_index", T.IntegerType()),
        T.StructField("source_time_in_s", T.DoubleType()),
        T.StructField("replay_event_time", T.StringType()),
        T.StructField("replay_sequence", T.LongType()),
        T.StructField("voltage_in_V", T.DoubleType()),
        T.StructField("current_in_A", T.DoubleType()),
        T.StructField("temperature_in_C", T.DoubleType()),
        T.StructField("discharge_capacity_in_Ah", T.DoubleType()),
        T.StructField("internal_resistance_in_ohm", T.DoubleType()),
        T.StructField("charge_capacity_in_Ah", T.DoubleType()),
        T.StructField("schema_version", T.StringType()),
    ]
)

LIFECYCLE_SCHEMA = T.StructType([
    T.StructField("event_id", T.StringType()),
    T.StructField("event_type", T.StringType()),
    T.StructField("dataset", T.StringType()),
    T.StructField("battery_id", T.StringType()),
    T.StructField("cycle_index", T.IntegerType()),
    T.StructField("replay_event_time", T.StringType()),
    T.StructField("replay_sequence", T.LongType()),
    T.StructField("expected_telemetry_rows", T.LongType()),
    T.StructField("schema_version", T.StringType()),
])

CURRENT_STREAM_STATE_SCHEMA = T.StructType([
    T.StructField("dataset", T.StringType(), False),
    T.StructField("state_id", T.StringType(), False),
    T.StructField("feature_contract_version", T.StringType(), False),
    T.StructField("published_at", T.StringType(), False),
])

SERVING_STATUS_SCHEMA = T.StructType([
    T.StructField("dataset", T.StringType(), False),
    T.StructField("state_id", T.StringType(), False),
    T.StructField("consumer", T.StringType(), False),
    T.StructField("selection_revision", T.IntegerType(), False),
    T.StructField("model_version", T.StringType(), True),
    T.StructField("model_fingerprint", T.StringType(), True),
    T.StructField("status", T.StringType(), False),
    T.StructField("rows_written", T.IntegerType(), False),
    T.StructField("error_message", T.StringType(), True),
    T.StructField("updated_at", T.StringType(), False),
])

CURRENT_PREDICTION_SCHEMA = T.StructType([
    T.StructField("dataset", T.StringType(), False),
    T.StructField("battery_id", T.StringType(), False),
    T.StructField("model_version", T.StringType(), False),
    T.StructField("model_fingerprint", T.StringType(), False),
    T.StructField("state_id", T.StringType(), False),
    T.StructField("replay_sequence", T.LongType(), False),
    T.StructField("cycle_index", T.IntegerType(), False),
    T.StructField("raw_predicted_rul_cycles", T.DoubleType(), False),
    T.StructField("predicted_rul_cycles", T.DoubleType(), False),
    T.StructField("predicted_eol_cycle", T.DoubleType(), False),
    T.StructField("inference_created_at", T.StringType(), False),
    T.StructField("selection_revision", T.IntegerType(), False),
])


def build_spark_session(master=DEFAULT_MASTER):
    try:
        from .spark_environment import configure_local_python
    except ImportError:
        from spark_environment import configure_local_python
    configure_local_python(master)
    """Create the small cluster session used by the streaming job."""
    return (
        SparkSession.builder.master(master)
        .appName("matr-battery-kafka-streaming")
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.cores.max", 2)
        .config("spark.executor.cores", 1)
        .config("spark.dynamicAllocation.enabled", "false")
        .getOrCreate()
    )


def _payload_with_kafka_metadata(kafka_records, schema):
    metadata = []
    for column, dtype in (("topic", "string"), ("partition", "int"), ("offset", "long")):
        metadata.append(F.col(column) if column in kafka_records.columns else F.lit(None).cast(dtype).alias(column))
    return kafka_records.select(F.from_json(F.col("value").cast("string"), schema).alias("payload"), *metadata)


def parse_telemetry(kafka_records):
    """Decode valid Kafka JSON telemetry and normalize its event time to UTC."""
    return (
        _payload_with_kafka_metadata(kafka_records, TELEMETRY_SCHEMA)
        .where(F.col("payload").isNotNull())
        .select("payload.*", "topic", "partition", "offset")
        .withColumn("event_time", F.to_timestamp("replay_event_time"))
        .where(F.col("event_id").isNotNull() & F.col("dataset").isNotNull() & F.col("battery_id").isNotNull() & F.col("cycle_index").isNotNull() & F.col("sample_index").isNotNull() & F.col("event_time").isNotNull() & (F.col("schema_version") == "1.0"))
    )


def parse_lifecycle(kafka_records):
    """Decode the immutable EOL and replay-completion facts."""
    return (
        _payload_with_kafka_metadata(kafka_records, LIFECYCLE_SCHEMA)
        .where(F.col("payload").isNotNull())
        .select("payload.*", "topic", "partition", "offset")
        .withColumn("event_time", F.to_timestamp("replay_event_time"))
        .where(F.col("event_id").isNotNull() & F.col("battery_id").isNotNull() & F.col("event_time").isNotNull() & F.col("event_type").isin("cycle_complete", "eol_observed", "replay_complete") & (F.col("schema_version") == "1.0"))
    )


def deduplicate_events(telemetry):
    """Apply the existing event-time bounded event-id deduplication."""
    if telemetry.isStreaming:
        return telemetry.withWatermark("event_time", WATERMARK_DELAY).dropDuplicatesWithinWatermark(["event_id"])
    return telemetry.dropDuplicates(["event_id"])


def _cycle_health_aggregates(telemetry):
    """Return additive aggregates so a micro-batch can be merged by cycle key."""
    return telemetry.groupBy("battery_id", "cycle_index").agg(*spark_cycle_aggregate_expressions(F, telemetry.columns))


def _render_cycle_health(aggregates):
    return render_spark_cycle_aggregate(aggregates, F)


def build_window_metrics(telemetry):
    """Build deterministic battery-cycle health aggregates for parity checks."""
    return _render_cycle_health(_cycle_health_aggregates(deduplicate_events(telemetry)))


def build_lifecycle_state(lifecycle):
    """Materialize EOL observation independently from source replay completion."""
    return deduplicate_events(lifecycle).groupBy("battery_id").agg(
        F.max(F.when(F.col("event_type") == "eol_observed", 1).otherwise(0)).cast("boolean").alias("eol_observed"),
        F.max(F.when(F.col("event_type") == "replay_complete", 1).otherwise(0)).cast("boolean").alias("replay_complete"),
        F.max(F.when(F.col("event_type") == "eol_observed", F.col("event_time"))).alias("eol_observed_at"),
        F.max(F.when(F.col("event_type") == "replay_complete", F.col("event_time"))).alias("replay_complete_at"),
    )


def finalized_cycle_boundary(lifecycle):
    """Return only explicitly completed cycles; no timestamp reconstruction is allowed."""
    return deduplicate_events(lifecycle).where(F.col("event_type") == "cycle_complete").select(
        "dataset", "battery_id", "cycle_index", "replay_sequence", "expected_telemetry_rows",
        "event_time", "topic", "partition", "offset"
    ).dropDuplicates(["dataset", "battery_id", "cycle_index"])


def upsert_cycle_health(batch, _batch_id, output_path):
    """Merge a micro-batch into canonical Parquet by (battery_id, cycle_index)."""
    if batch.rdd.isEmpty():
        return
    output_path = Path(output_path)
    incoming = _cycle_health_aggregates(batch)
    if output_path.exists():
        incoming = incoming.unionByName(batch.sparkSession.read.parquet(str(output_path)), allowMissingColumns=True)
    merged = incoming.groupBy("battery_id", "cycle_index").agg(
        F.sum("event_count").alias("event_count"),
        F.sum("_voltage_sum").alias("_voltage_sum"), F.sum("_voltage_count").alias("_voltage_count"),
        F.sum("_current_sum").alias("_current_sum"), F.sum("_current_count").alias("_current_count"),
        F.max("charge_time_in_s").alias("charge_time_in_s"), F.max("charge_capacity_in_Ah").alias("charge_capacity_in_Ah"),
        F.max("discharge_capacity_in_Ah").alias("discharge_capacity_in_Ah"), F.max("internal_resistance_in_ohm").alias("internal_resistance_in_ohm"),
        F.min("temperature_min_in_C").alias("temperature_min_in_C"), F.min("voltage_min_in_V").alias("voltage_min_in_V"),
        F.max("temperature_max_in_C").alias("temperature_max_in_C"), F.max("voltage_max_in_V").alias("voltage_max_in_V"),
        F.max("current_abs_max_in_A").alias("current_abs_max_in_A"),
    )
    health = _render_cycle_health(merged).cache()
    try:
        health.count()  # Materialize before replacing the Parquet snapshot it read.
        # ponytail: snapshot rewrite is bounded-replay scale; use a transactional table if this grows.
        health.write.mode("overwrite").parquet(str(output_path))
    finally:
        health.unpersist()


def upsert_lifecycle_state(batch, _batch_id, output_path):
    """Merge immutable lifecycle facts into one state row per battery."""
    if batch.rdd.isEmpty():
        return
    output_path = Path(output_path)
    incoming = build_lifecycle_state(batch)
    if output_path.exists():
        incoming = incoming.unionByName(batch.sparkSession.read.parquet(str(output_path)), allowMissingColumns=True)
    state = incoming.groupBy("battery_id").agg(
        F.max(F.col("eol_observed").cast("int")).cast("boolean").alias("eol_observed"),
        F.max(F.col("replay_complete").cast("int")).cast("boolean").alias("replay_complete"),
        F.max("eol_observed_at").alias("eol_observed_at"),
        F.max("replay_complete_at").alias("replay_complete_at"),
    )
    state.write.mode("overwrite").parquet(str(output_path))


def upsert_completed_cycles(batch, output_path):
    """Persist only explicit cycle completion facts for the authoritative boundary."""
    incoming = finalized_cycle_boundary(batch)
    if incoming.rdd.isEmpty():
        return
    output_path = Path(output_path)
    if output_path.exists():
        incoming = incoming.unionByName(batch.sparkSession.read.parquet(str(output_path)))
    incoming.dropDuplicates(["dataset", "battery_id", "cycle_index"]).write.mode("overwrite").parquet(str(output_path))


def upsert_kafka_offset_watermarks(batch, output_path):
    """Persist inclusive source offsets independently of mutable Spark checkpoints."""
    incoming = batch.select("topic", "partition", "offset").where(
        F.col("topic").isin(TOPIC, LIFECYCLE_TOPIC) & F.col("partition").isNotNull() & F.col("offset").isNotNull()
    ).groupBy("topic", "partition").agg(F.max("offset").alias("offset"))
    if incoming.rdd.isEmpty():
        return
    output_path = Path(output_path)
    if output_path.exists():
        incoming = incoming.unionByName(batch.sparkSession.read.parquet(str(output_path)))
    incoming.groupBy("topic", "partition").agg(F.max("offset").alias("offset")).write.mode("overwrite").parquet(str(output_path))


def kafka_offset_watermarks(spark, path):
    """Return canonical inclusive high-watermarks by Kafka topic and partition."""
    path = Path(path)
    if not path.exists():
        return {}
    result = {}
    for row in spark.read.parquet(str(path)).orderBy("topic", "partition").collect():
        result.setdefault(row.topic, {})[str(int(row.partition))] = int(row.offset)
    return result


def shared_training_cohort(arrival_rows, *, arrived_battery_ids, observed_eol_ids):
    """Bind both model families to one arrived training cohort."""
    arrived_battery_ids, observed_eol_ids = set(arrived_battery_ids), set(observed_eol_ids)
    arrived = [
        row["battery_id"] for row in sorted(arrival_rows, key=lambda row: row["arrival_rank"])
        if row["split"] == "train" and row["battery_id"] in arrived_battery_ids
    ]
    observed = [battery_id for battery_id in arrived if battery_id in observed_eol_ids]
    observed_set = set(observed)
    return {
        "arrived_train_battery_ids": arrived,
        "observed_eol_train_battery_ids": observed,
        "censored_train_battery_ids": [battery_id for battery_id in arrived if battery_id not in observed_set],
    }


def publish_completed_state(spark, *, health_path, completed_cycles_path, offset_watermarks_path, state_root, canonical_fingerprint, arrival_manifest_fingerprint=None, lifecycle_path, batch_id, arrival_manifest_path=DEFAULT_ARRIVAL_MANIFEST_PATH, canonical_cycles_path=DEFAULT_CANONICAL_CYCLES_PATH):
    """Publish state only after both the accumulated health and completion boundary exist."""
    if not Path(health_path).exists() or not Path(completed_cycles_path).exists():
        return None
    completed = spark.read.parquet(str(completed_cycles_path))
    if "expected_telemetry_rows" not in completed.columns:
        raise RuntimeError("cycle completion facts require expected_telemetry_rows")
    health = spark.read.parquet(str(health_path))
    counts = health.select("battery_id", "cycle_index", "event_count")
    completion = completed.join(counts, ["battery_id", "cycle_index"], "left")
    if completion.where(F.col("event_count") > F.col("expected_telemetry_rows")).limit(1).count():
        raise RuntimeError("cycle telemetry exceeds its immutable completion count")
    ready = completion.where(
        F.col("expected_telemetry_rows").isNotNull()
        & (F.col("event_count") == F.col("expected_telemetry_rows"))
    ).orderBy("dataset", "battery_id", "cycle_index")
    ready_rows = [row.asDict() for row in ready.collect()]
    canonical_rows = [row.asDict() for row in spark.read.parquet(str(canonical_cycles_path)).select(
        "dataset", "battery_id", "cycle_index"
    ).orderBy("dataset", "battery_id", "cycle_index").collect()]
    ready_keys = {(row["dataset"], row["battery_id"], int(row["cycle_index"])) for row in ready_rows}
    prefix_keys = set()
    blocked = set()
    for row in canonical_rows:
        key = row["dataset"], row["battery_id"], int(row["cycle_index"])
        battery = key[:2]
        if battery in blocked:
            continue
        if key in ready_keys:
            prefix_keys.add(key)
        else:
            blocked.add(battery)
    prefix_rows = [row for row in ready_rows if (row["dataset"], row["battery_id"], int(row["cycle_index"])) in prefix_keys]
    if not prefix_rows:
        return None
    arrival_rows = [row.asDict() for row in spark.read.parquet(str(arrival_manifest_path)).select(
        "battery_id", "split", "arrival_rank", "schedule_fingerprint"
    ).collect()]
    arrival_fingerprints = {row["schedule_fingerprint"] for row in arrival_rows}
    if len(arrival_fingerprints) != 1:
        raise RuntimeError("arrival manifest must contain one schedule fingerprint")
    actual_arrival_fingerprint = arrival_fingerprints.pop()
    if arrival_manifest_fingerprint and arrival_manifest_fingerprint != actual_arrival_fingerprint:
        raise RuntimeError("configured arrival manifest fingerprint does not match the replay manifest")
    boundary = build_compact_finalized_cycle_boundary(
        prefix_rows, canonical_cycle_keys=canonical_rows, canonical_fingerprint=canonical_fingerprint,
        arrival_manifest_fingerprint=actual_arrival_fingerprint, feature_contract_version=FEATURE_VERSION,
    )
    state_id = state_id_for_boundary(boundary)
    if (Path(state_root) / "stream_state" / state_id / "manifest.json").exists():
        return None
    kafka_offsets = kafka_offset_watermarks(spark, offset_watermarks_path)
    if not kafka_offsets:
        raise RuntimeError("Kafka-produced finalized state requires source offset watermarks")
    if not {"topic", "partition", "offset"}.issubset(ready.columns):
        raise RuntimeError("finalized cycle completion facts require Kafka source metadata")
    for row in ready.select("topic", "partition", "offset").collect():
        if kafka_offsets.get(row.topic, {}).get(str(int(row.partition)), -1) < int(row.offset):
            raise RuntimeError("finalized cycle completion exceeds Kafka source watermark")
    prefix = spark.createDataFrame(prefix_rows).select("battery_id", "cycle_index", "replay_sequence", "event_time")
    state = health.join(prefix, ["battery_id", "cycle_index"])
    outlet_path = Path(state_root) / "shared_feature_outlet"
    maxima = feature_outlet_key_maxima(outlet_path) if (outlet_path / "_outlet.json").exists() else {}
    new_keys = {(row["dataset"], row["battery_id"], int(row["cycle_index"])) for row in prefix_rows
                if int(row["cycle_index"]) > maxima.get((row["dataset"], row["battery_id"]), 0)}
    context_keys = set()
    prefix_by_battery = {}
    for row in prefix_rows:
        prefix_by_battery.setdefault((row["dataset"], row["battery_id"]), []).append(int(row["cycle_index"]))
    for dataset_battery, cycles in prefix_by_battery.items():
        cycles.sort()
        for cycle in (value for value in cycles if (*dataset_battery, value) in new_keys):
            position = cycles.index(cycle)
            context_keys.update((*dataset_battery, value) for value in ({cycles[0]} | set(cycles[max(0, position - 9):position + 1])))
    if context_keys:
        context = spark.createDataFrame([
            {"battery_id": battery_id, "cycle_index": cycle_index}
            for _, battery_id, cycle_index in sorted(context_keys)
        ]).join(state, ["battery_id", "cycle_index"])
        context_rows = [{**row.asDict(), "dataset": "MATR"} for row in context.collect()]
        causal_rows = [row for row in feature_rows(context_rows)
                       if (row["dataset"], row["battery_id"], int(row["cycle_index"])) in new_keys]
    else:
        causal_rows = []
    lifecycle = spark.read.parquet(str(lifecycle_path)).where("eol_observed AND replay_complete").select("battery_id").collect() if Path(lifecycle_path).exists() else []
    if "event_time" not in ready.columns:
        raise RuntimeError("finalized cycle completion facts require replay event time for the training cutoff")
    cutoff = prefix.select(
        F.date_format(F.max("event_time"), "yyyy-MM-dd'T'HH:mm:ssXXX").alias("replay_cutoff")
    ).first().replay_cutoff
    if cutoff is None:
        raise RuntimeError("finalized cycle completion facts require a replay cutoff")
    cohort = shared_training_cohort(
        arrival_rows,
        arrived_battery_ids={row["battery_id"] for row in prefix_rows},
        observed_eol_ids={row.battery_id for row in lifecycle},
    )
    by_generation = {}
    for row in causal_rows:
        by_generation.setdefault(generation_for_timestamp(row["event_time"]), []).append({
            key: row.get(key) for key in ("dataset", "battery_id", "cycle_index", *SHARED_FEATURE_COLUMNS)
            if key in row
        })
    for generation, rows in sorted(by_generation.items()):
        append_shared_feature_rows(
            outlet_path, rows, generation=generation,
            feature_contract_version=FEATURE_VERSION, canonical_source_fingerprint=canonical_fingerprint,
        )
    return publish_state_artifacts(
        state_root, finalized_keys=[{
            "dataset": row["dataset"], "battery_id": row["battery_id"],
            "cycle_index": int(row["cycle_index"]),
        } for row in prefix_rows], state_rows=[], feature_rows=causal_rows,
        canonical_fingerprint=canonical_fingerprint, arrival_manifest_fingerprint=actual_arrival_fingerprint,
        feature_contract_version=FEATURE_VERSION,
        eligible_completed_training_batteries=cohort["observed_eol_train_battery_ids"],
        cutoff_metadata={"batch_id": int(batch_id), "replay_cutoff": cutoff},
        kafka_offsets=kafka_offsets, require_kafka_offsets=True, shared_training_cohort=cohort,
        boundary=boundary,
    )


def record_current_stream_state(spark, state_root, manifest):
    """Mirror the already-published finalized state into PostgreSQL for Streamlit."""
    _, url, props = _serving_selection(spark, "current_models", "model_evaluations")
    stage = "analytics.stream_state_" + re.sub("[^a-z0-9_]", "_", manifest["state_id"][-16:].lower())
    spark.createDataFrame([current_stream_state_row("MATR", manifest)], CURRENT_STREAM_STATE_SCHEMA).withColumn("published_at", F.to_timestamp("published_at")).write.jdbc(url, stage, mode="overwrite", properties=props)
    _execute(spark, url, props["user"], props["password"], build_current_stream_state_upsert_sql(stage))
    status_stage = stage + "_status"
    pending = [serving_status_row("MATR", manifest["state_id"], consumer, None, status="pending") for consumer in ("rul_current", "survival_current")]
    spark.createDataFrame(pending, SERVING_STATUS_SCHEMA).withColumn("updated_at", F.to_timestamp("updated_at")).write.jdbc(url, status_stage, mode="overwrite", properties=props)
    _execute(spark, url, props["user"], props["password"], build_serving_status_upsert_sql(status_stage))


def record_rul_status(spark, manifest, result=None, error=None):
    selection, url, props = _serving_selection(spark, "current_models", "model_evaluations")
    state = re.sub("[^a-z0-9_]", "_", manifest["state_id"][-16:].lower())
    stage = "analytics.stream_rul_status_" + state
    status = "failed" if error else ("unavailable" if result and result.get("status") == "no_current_model" else "served")
    row = serving_status_row("MATR", manifest["state_id"], "rul_current", selection, status=status,
        rows_written=(result or {}).get("rows", 0), error_message=str(error)[:500] if error else None)
    spark.createDataFrame([row], SERVING_STATUS_SCHEMA).withColumn("updated_at", F.to_timestamp("updated_at")).write.jdbc(url, stage, mode="overwrite", properties=props)
    _execute(spark, url, props["user"], props["password"], build_serving_status_upsert_sql(stage))


def _serving_selection(spark, table, evaluation_table):
    url = os.environ.get("POSTGRES_JDBC_URL", "jdbc:postgresql://localhost:5432/battery_reliability")
    props = jdbc_properties(os.environ.get("POSTGRES_USER", "battery_reliability"), os.environ["POSTGRES_PASSWORD"])
    selected_fingerprint = ", current.model_fingerprint AS selected_fingerprint" if table == "current_survival_models" else ""
    query = f"(SELECT current.dataset, current.model_version, current.selection_revision, evaluation.model_fingerprint, evaluation.training_metadata{selected_fingerprint} FROM analytics.{table} current JOIN analytics.{evaluation_table} evaluation USING (model_version)) selection"
    rows = spark.read.jdbc(url, query, properties=props).collect()
    return (rows[0].asDict(), url, props) if rows else (None, url, props)


def _latest_features(state_root, manifest, benchmark_battery_ids=()):
    return load_current_feature_rows(state_root, manifest, excluded_battery_ids=benchmark_battery_ids)


def run_current_rul_inference(spark, state_root, manifest):
    """Pin one PostgreSQL selection and monotonically merge newest-cycle RUL rows."""
    selection, url, props = _serving_selection(spark, "current_models", "model_evaluations")
    if selection is None:
        return {"status": "no_current_model"}
    import joblib
    metadata = selection["training_metadata"] if isinstance(selection["training_metadata"], dict) else json.loads(selection["training_metadata"])
    if metadata.get("feature_version") != manifest["feature_contract_version"]:
        raise RuntimeError("current RUL model feature contract mismatch")
    model = joblib.load(Path(state_root) / "model_generations" / selection["model_fingerprint"] / "selected_model.joblib")
    benchmark = json.loads((Path(state_root) / "fixed_offline_benchmark/v1/benchmark.json").read_text())
    excluded = set(benchmark["splits"]["validation"]["battery_ids"]) | set(benchmark["splits"]["test"]["battery_ids"])
    features = _latest_features(state_root, manifest, excluded)
    prior = {row.battery_id: row.asDict() for row in spark.read.jdbc(url, "analytics.battery_current_predictions", properties=props).collect()}
    rows = current_prediction_rows(model, features, feature_columns=RUL_FEATURES, model_version=selection["model_version"], model_fingerprint=selection["model_fingerprint"], state_id=manifest["state_id"], selection_revision=selection["selection_revision"], benchmark_battery_ids=excluded, prior_predictions=prior)
    if not rows:
        return {"status": "no_eligible_features"}
    stage = "analytics.stream_rul_" + re.sub("[^a-z0-9_]", "_", manifest["state_id"][-16:].lower())
    spark.createDataFrame(rows, CURRENT_PREDICTION_SCHEMA).withColumn("inference_created_at", F.to_timestamp("inference_created_at")).write.jdbc(url, stage, mode="overwrite", properties=props)
    _execute(spark, url, props["user"], props["password"], build_current_prediction_upsert_sql(stage))
    return {"status": "served", "rows": len(rows)}


def start_query(spark, output_path, checkpoint_path, lifecycle_output_path=DEFAULT_LIFECYCLE_OUTPUT_PATH, completed_cycles_path=DEFAULT_COMPLETED_CYCLES_PATH, offset_watermarks_path=DEFAULT_KAFKA_OFFSETS_PATH, state_root=DEFAULT_STATE_ROOT, canonical_fingerprint="matr-canonical-v1", arrival_manifest_fingerprint=None, available_now=False, arrival_manifest_path=DEFAULT_ARRIVAL_MANIFEST_PATH, canonical_cycles_path=DEFAULT_CANONICAL_CYCLES_PATH):
    """Start the checkpointed Kafka query and return its StreamingQuery."""
    kafka_records = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", DEFAULT_BOOTSTRAP_SERVER)
        .option("subscribe", f"{TOPIC},{LIFECYCLE_TOPIC}")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
        .load()
    )
    def process_batch(batch, batch_id):
        upsert_kafka_offset_watermarks(batch, offset_watermarks_path)
        upsert_cycle_health(deduplicate_events(parse_telemetry(batch)), batch_id, output_path)
        lifecycle = deduplicate_events(parse_lifecycle(batch))
        upsert_lifecycle_state(lifecycle, batch_id, lifecycle_output_path)
        upsert_completed_cycles(lifecycle, completed_cycles_path)
        manifest = publish_completed_state(spark, health_path=output_path, completed_cycles_path=completed_cycles_path,
            offset_watermarks_path=offset_watermarks_path, state_root=state_root, canonical_fingerprint=canonical_fingerprint,
            arrival_manifest_fingerprint=arrival_manifest_fingerprint, lifecycle_path=lifecycle_output_path,
            arrival_manifest_path=arrival_manifest_path, canonical_cycles_path=canonical_cycles_path, batch_id=batch_id)
        if manifest:
            # latest.json is finalized state only; serving failures cannot retract it.
            try:
                record_current_stream_state(spark, state_root, manifest)
            except Exception as error:
                print(f"current stream-state mirror failed after finalized state publication: {error}", flush=True)
            try:
                result = run_current_rul_inference(spark, state_root, manifest)
                record_rul_status(spark, manifest, result)
            except Exception as error:
                print(f"current RUL serving failed after finalized state publication: {error}", flush=True)
                try:
                    record_rul_status(spark, manifest, error=error)
                except Exception:
                    pass

    writer = (
        kafka_records.writeStream.foreachBatch(process_batch)
        .outputMode("append")
        .option("checkpointLocation", str(checkpoint_path))
    )
    return writer.trigger(availableNow=True).start() if available_now else writer.trigger(processingTime="5 seconds").start()


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Kafka telemetry with Structured Streaming.")
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--lifecycle-output-path", type=Path, default=DEFAULT_LIFECYCLE_OUTPUT_PATH)
    parser.add_argument("--completed-cycles-path", type=Path, default=DEFAULT_COMPLETED_CYCLES_PATH)
    parser.add_argument("--offset-watermarks-path", type=Path, default=DEFAULT_KAFKA_OFFSETS_PATH)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--canonical-fingerprint", default="matr-canonical-v1")
    parser.add_argument("--arrival-manifest-fingerprint", help="optional expected fingerprint; the state records the manifest's actual fingerprint")
    parser.add_argument("--arrival-manifest-path", type=Path, default=DEFAULT_ARRIVAL_MANIFEST_PATH)
    parser.add_argument("--canonical-cycles-path", type=Path, default=DEFAULT_CANONICAL_CYCLES_PATH)
    parser.add_argument("--available-now", action="store_true", help="Process current offsets in bounded micro-batches, then exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = build_spark_session(args.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        query = start_query(spark, args.output_path, args.checkpoint_path, args.lifecycle_output_path, args.completed_cycles_path,
            args.offset_watermarks_path, args.state_root, args.canonical_fingerprint, args.arrival_manifest_fingerprint,
            args.available_now, args.arrival_manifest_path, args.canonical_cycles_path)
        query.awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
