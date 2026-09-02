"""Aggregate Kafka telemetry with PySpark Structured Streaming."""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T


DEFAULT_MASTER = "spark://spark-master:7077"
DEFAULT_BOOTSTRAP_SERVER = "kafka:29092"
DEFAULT_OUTPUT_PATH = Path("data/processed/matr/replay_cycle_health")
DEFAULT_CHECKPOINT_PATH = Path("data/processed/matr/replay_checkpoint")
DEFAULT_LIFECYCLE_OUTPUT_PATH = Path("data/processed/matr/replay_lifecycle_state")
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
    T.StructField("schema_version", T.StringType()),
])


def build_spark_session(master=DEFAULT_MASTER):
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


def parse_telemetry(kafka_records):
    """Decode valid Kafka JSON telemetry and normalize its event time to UTC."""
    payload = F.from_json(F.col("value").cast("string"), TELEMETRY_SCHEMA)
    return (
        kafka_records.select(payload.alias("payload"))
        .where(F.col("payload").isNotNull())
        .select("payload.*")
        .withColumn("event_time", F.to_timestamp("replay_event_time"))
        .where(F.col("event_id").isNotNull() & F.col("dataset").isNotNull() & F.col("battery_id").isNotNull() & F.col("cycle_index").isNotNull() & F.col("sample_index").isNotNull() & F.col("event_time").isNotNull() & (F.col("schema_version") == "1.0"))
    )


def parse_lifecycle(kafka_records):
    """Decode the immutable EOL and replay-completion facts."""
    payload = F.from_json(F.col("value").cast("string"), LIFECYCLE_SCHEMA)
    return (
        kafka_records.select(payload.alias("payload"))
        .where(F.col("payload").isNotNull())
        .select("payload.*")
        .withColumn("event_time", F.to_timestamp("replay_event_time"))
        .where(F.col("event_id").isNotNull() & F.col("battery_id").isNotNull() & F.col("event_time").isNotNull() & F.col("event_type").isin("eol_observed", "replay_complete") & (F.col("schema_version") == "1.0"))
    )


def deduplicate_events(telemetry):
    """Apply the existing event-time bounded event-id deduplication."""
    if telemetry.isStreaming:
        return telemetry.withWatermark("event_time", WATERMARK_DELAY).dropDuplicatesWithinWatermark(["event_id"])
    return telemetry.dropDuplicates(["event_id"])


def _cycle_health_aggregates(telemetry):
    """Return additive aggregates so a micro-batch can be merged by cycle key."""
    def optional_maximum(column):
        return F.max(column) if column in telemetry.columns else F.lit(None).cast("double")

    return telemetry.groupBy("battery_id", "cycle_index").agg(
        F.count("*").alias("event_count"),
        F.sum("voltage_in_V").alias("_voltage_sum"),
        F.count("voltage_in_V").alias("_voltage_count"),
        optional_maximum("temperature_in_C").alias("maximum_temperature_in_C"),
        optional_maximum("charge_capacity_in_Ah").alias("charge_capacity_in_Ah"),
        optional_maximum("discharge_capacity_in_Ah").alias("discharge_capacity_in_Ah"),
        optional_maximum("internal_resistance_in_ohm").alias("internal_resistance_in_ohm"),
    )


def _render_cycle_health(aggregates):
    return aggregates.withColumn(
        "average_voltage_in_V",
        F.when(F.col("_voltage_count") > 0, F.col("_voltage_sum") / F.col("_voltage_count")),
    )


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
        F.sum("_voltage_sum").alias("_voltage_sum"),
        F.sum("_voltage_count").alias("_voltage_count"),
        F.max("maximum_temperature_in_C").alias("maximum_temperature_in_C"),
        F.max("charge_capacity_in_Ah").alias("charge_capacity_in_Ah"),
        F.max("discharge_capacity_in_Ah").alias("discharge_capacity_in_Ah"),
        F.max("internal_resistance_in_ohm").alias("internal_resistance_in_ohm"),
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
        incoming = incoming.unionByName(batch.sparkSession.read.parquet(str(output_path)))
    state = incoming.groupBy("battery_id").agg(
        F.max(F.col("eol_observed").cast("int")).cast("boolean").alias("eol_observed"),
        F.max(F.col("replay_complete").cast("int")).cast("boolean").alias("replay_complete"),
        F.max("eol_observed_at").alias("eol_observed_at"),
        F.max("replay_complete_at").alias("replay_complete_at"),
    )
    state.write.mode("overwrite").parquet(str(output_path))


def start_query(spark, output_path, checkpoint_path, lifecycle_output_path=DEFAULT_LIFECYCLE_OUTPUT_PATH, available_now=False):
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
        upsert_cycle_health(deduplicate_events(parse_telemetry(batch)), batch_id, output_path)
        upsert_lifecycle_state(deduplicate_events(parse_lifecycle(batch)), batch_id, lifecycle_output_path)

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
    parser.add_argument("--available-now", action="store_true", help="Process current offsets in bounded micro-batches, then exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = build_spark_session(args.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        query = start_query(spark, args.output_path, args.checkpoint_path, args.lifecycle_output_path, args.available_now)
        query.awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
