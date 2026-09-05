import shutil
import subprocess
import sys

import pytest

from src.spark_streaming import (
    build_lifecycle_state,
    build_window_metrics,
    kafka_offset_watermarks,
    parse_lifecycle,
    parse_telemetry,
    publish_completed_state,
    upsert_cycle_health,
    upsert_kafka_offset_watermarks,
)


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None or subprocess.run(["java", "-version"], capture_output=True).returncode:
        pytest.skip("Java 17 is required to run local Spark tests")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.master("local[1]").config("spark.pyspark.python", sys.executable).appName("spark-streaming-test").getOrCreate()
    yield session
    session.stop()


def test_window_metrics_excludes_malformed_json_and_duplicate_event_ids(spark):
    kafka_records = spark.createDataFrame(
        [
            ('{"event_id":"one","dataset":"MATR","battery_id":"MATR-1","cycle_index":1,"sample_index":0,"replay_event_time":"2020-01-01T00:00:00","voltage_in_V":3.2,"temperature_in_C":25.0,"schema_version":"1.0"}',),
            ('{"event_id":"one","dataset":"MATR","battery_id":"MATR-1","cycle_index":1,"sample_index":0,"replay_event_time":"2020-01-01T00:00:00","voltage_in_V":3.2,"temperature_in_C":25.0,"schema_version":"1.0"}',),
            ('{"event_id":"two","dataset":"MATR","battery_id":"MATR-1","cycle_index":1,"sample_index":1,"replay_event_time":"2020-01-01T01:00:00","voltage_in_V":3.15,"temperature_in_C":27.0,"schema_version":"1.0"}',),
            ("not-json",),
        ],
        "value string",
    )

    rows = build_window_metrics(parse_telemetry(kafka_records)).collect()

    assert len(rows) == 1
    assert rows[0].battery_id == "MATR-1"
    assert rows[0].cycle_index == 1
    assert rows[0].event_count == 2
    assert rows[0].average_voltage_in_V == 3.175
    assert rows[0].maximum_temperature_in_C == 27.0


def test_upsert_cycle_health_merges_the_same_natural_key(spark, tmp_path):
    rows = spark.createDataFrame([("MATR-1", 1, 3.0, 25.0), ("MATR-1", 1, 3.2, 27.0)], "battery_id string, cycle_index int, voltage_in_V double, temperature_in_C double")
    upsert_cycle_health(rows, 0, tmp_path / "health")
    upsert_cycle_health(spark.createDataFrame([("MATR-1", 1, 3.4, 26.0)], "battery_id string, cycle_index int, voltage_in_V double, temperature_in_C double"), 1, tmp_path / "health")
    result = spark.read.parquet(str(tmp_path / "health")).collect()
    assert len(result) == 1
    assert result[0].event_count == 3
    assert result[0].average_voltage_in_V == pytest.approx(3.2)


def test_lifecycle_state_keeps_completion_separate_from_eol_observation(spark):
    records = spark.createDataFrame([
        ('{"event_id":"complete","event_type":"replay_complete","dataset":"MATR","battery_id":"unverified","cycle_index":4,"replay_event_time":"2020-01-04T00:00:00","schema_version":"1.0"}',),
        ('{"event_id":"eol","event_type":"eol_observed","dataset":"MATR","battery_id":"valid","cycle_index":3,"replay_event_time":"2020-01-03T00:00:00","schema_version":"1.0"}',),
        ('{"event_id":"valid-complete","event_type":"replay_complete","dataset":"MATR","battery_id":"valid","cycle_index":4,"replay_event_time":"2020-01-04T00:00:00","schema_version":"1.0"}',),
    ], "value string")

    state = {row.battery_id: row for row in build_lifecycle_state(parse_lifecycle(records)).collect()}

    assert (state["unverified"].replay_complete, state["unverified"].eol_observed) == (True, False)
    assert (state["valid"].replay_complete, state["valid"].eol_observed) == (True, True)


def test_kafka_metadata_survives_parsing_and_watermarks_are_monotonic(spark, tmp_path):
    records = spark.createDataFrame([
        ("battery_measurements", 0, 4, '{"event_id":"one","dataset":"MATR","battery_id":"b","cycle_index":1,"sample_index":0,"replay_event_time":"2020-01-01T00:00:00","schema_version":"1.0"}'),
        ("battery_lifecycle", 1, 7, '{"event_id":"complete","event_type":"cycle_complete","dataset":"MATR","battery_id":"b","cycle_index":1,"replay_event_time":"2020-01-01T00:00:01","schema_version":"1.0"}'),
    ], "topic string, partition int, offset long, value string")

    assert parse_telemetry(records).where("event_id = 'one'").collect()[0].offset == 4
    assert parse_lifecycle(records).where("event_id = 'complete'").collect()[0].partition == 1
    upsert_kafka_offset_watermarks(records, tmp_path / "offsets")
    upsert_kafka_offset_watermarks(spark.createDataFrame([("battery_measurements", 0, 9, "{}")], records.schema), tmp_path / "offsets")

    assert kafka_offset_watermarks(spark, tmp_path / "offsets") == {
        "battery_lifecycle": {"1": 7}, "battery_measurements": {"0": 9}
    }


def test_published_state_records_offsets_covering_finalized_completion(spark, tmp_path):
    health = tmp_path / "health"
    completed = tmp_path / "completed"
    lifecycle = tmp_path / "lifecycle"
    offsets = tmp_path / "offsets"
    spark.createDataFrame([("b", 1, 1, 3.2, 25.0, 1.1, 1.0, 0.01)],
        "battery_id string, cycle_index int, event_count long, average_voltage_in_V double, maximum_temperature_in_C double, charge_capacity_in_Ah double, discharge_capacity_in_Ah double, internal_resistance_in_ohm double").write.parquet(str(health))
    spark.createDataFrame([("MATR", "b", 1, 7, "battery_lifecycle", 1, 7)],
        "dataset string, battery_id string, cycle_index int, replay_sequence long, topic string, partition int, offset long").write.parquet(str(completed))
    spark.createDataFrame([], "battery_id string, eol_observed boolean, replay_complete boolean").write.parquet(str(lifecycle))
    upsert_kafka_offset_watermarks(spark.createDataFrame([
        ("battery_measurements", 0, 9, "{}"), ("battery_lifecycle", 1, 7, "{}")
    ], "topic string, partition int, offset long, value string"), offsets)

    manifest = publish_completed_state(
        spark, health_path=health, completed_cycles_path=completed, offset_watermarks_path=offsets,
        state_root=tmp_path, canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        lifecycle_path=lifecycle, batch_id=1,
    )

    assert manifest["kafka_offsets"]["battery_lifecycle"]["1"] >= 7
    assert manifest["kafka_offsets"]["battery_measurements"]["0"] == 9
    assert publish_completed_state(
        spark, health_path=health, completed_cycles_path=completed, offset_watermarks_path=offsets,
        state_root=tmp_path, canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        lifecycle_path=lifecycle, batch_id=2,
    ) is None
