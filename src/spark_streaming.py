"""Aggregate Kafka telemetry with PySpark Structured Streaming."""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T


DEFAULT_MASTER = "spark://spark-master:7077"
DEFAULT_BOOTSTRAP_SERVER = "kafka:29092"
DEFAULT_OUTPUT_PATH = Path("data/processed/spark_streaming_windows")
DEFAULT_CHECKPOINT_PATH = Path("data/processed/spark_streaming_checkpoint")
TOPIC = "vehicle_telemetry"
SHUFFLE_PARTITIONS = 3
MAX_OFFSETS_PER_TRIGGER = 100
WATERMARK_DELAY = "2 hours"
WINDOW_DURATION = "6 hours"

TELEMETRY_SCHEMA = T.StructType(
    [
        T.StructField("event_id", T.StringType()),
        T.StructField("vehicle_id", T.StringType()),
        T.StructField("timestamp", T.StringType()),
        T.StructField("battery_age_days", T.IntegerType()),
        T.StructField("battery_type", T.StringType()),
        T.StructField("region", T.StringType()),
        T.StructField("soc", T.DoubleType()),
        T.StructField("pack_voltage", T.DoubleType()),
        T.StructField("pack_current", T.DoubleType()),
        T.StructField("module_temp_min", T.DoubleType()),
        T.StructField("module_temp_max", T.DoubleType()),
        T.StructField("outside_temp", T.DoubleType()),
        T.StructField("odometer", T.DoubleType()),
        T.StructField("is_charging", T.BooleanType()),
    ]
)


def build_spark_session(master=DEFAULT_MASTER):
    """Create the small cluster session used by the streaming job."""
    return (
        SparkSession.builder.master(master)
        .appName("ev-fleet-kafka-streaming")
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
        .withColumn("event_time", F.to_timestamp("timestamp"))
        .where(F.col("event_id").isNotNull() & F.col("vehicle_id").isNotNull() & F.col("event_time").isNotNull())
    )


def build_window_metrics(telemetry):
    """Deduplicate telemetry and aggregate six-hour vehicle windows."""
    if telemetry.isStreaming:
        telemetry = telemetry.withWatermark("event_time", WATERMARK_DELAY).dropDuplicatesWithinWatermark(["event_id"])
    else:
        telemetry = telemetry.dropDuplicates(["event_id"])
    return telemetry.groupBy(F.window("event_time", WINDOW_DURATION).alias("window"), "vehicle_id").agg(
        F.count("*").alias("event_count"),
        F.avg("pack_voltage").alias("average_pack_voltage"),
        F.max("module_temp_max").alias("maximum_module_temperature"),
    )


def start_query(spark, output_path, checkpoint_path, available_now=False):
    """Start the checkpointed Kafka query and return its StreamingQuery."""
    kafka_records = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", DEFAULT_BOOTSTRAP_SERVER)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)
        .load()
    )
    writer = (
        build_window_metrics(parse_telemetry(kafka_records)).writeStream.format("parquet").outputMode("append")
        .option("path", str(output_path))
        .option("checkpointLocation", str(checkpoint_path))
    )
    return writer.trigger(availableNow=True).start() if available_now else writer.trigger(processingTime="5 seconds").start()


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Kafka telemetry with Structured Streaming.")
    parser.add_argument("--master", default=DEFAULT_MASTER)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--available-now", action="store_true", help="Process current offsets in bounded micro-batches, then exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = build_spark_session(args.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        query = start_query(spark, args.output_path, args.checkpoint_path, args.available_now)
        query.awaitTermination()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
