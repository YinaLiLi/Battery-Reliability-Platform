import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from src.spark_streaming import (
    build_lifecycle_state,
    build_window_metrics,
    kafka_offset_watermarks,
    parse_lifecycle,
    parse_telemetry,
    publish_completed_state,
    shared_training_cohort,
    upsert_cycle_health,
    upsert_kafka_offset_watermarks,
)


def test_shared_training_cohort_uses_arrived_train_members_and_event_semantics():
    cohort = shared_training_cohort(
        [{"battery_id": "late", "split": "train", "arrival_rank": 2},
         {"battery_id": "observed", "split": "train", "arrival_rank": 1},
         {"battery_id": "held-out", "split": "validation", "arrival_rank": 0}],
        arrived_battery_ids={"observed", "late", "held-out"},
        observed_eol_ids={"observed"},
    )

    assert cohort == {
        "arrived_train_battery_ids": ["observed", "late"],
        "observed_eol_train_battery_ids": ["observed"],
        "censored_train_battery_ids": ["late"],
    }


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
    arrival = tmp_path / "arrival"
    canonical = tmp_path / "cycle_summary"
    spark.createDataFrame([("b", 1, 1, 3.2, 3.1, 3.3, -1.0, 1.0, 21.0, 25.0, 1.1, 1.0, 0.01, 100.0)],
        "battery_id string, cycle_index int, event_count long, voltage_mean_in_V double, voltage_min_in_V double, voltage_max_in_V double, current_mean_in_A double, current_abs_max_in_A double, temperature_min_in_C double, temperature_max_in_C double, charge_capacity_in_Ah double, discharge_capacity_in_Ah double, internal_resistance_in_ohm double, charge_time_in_s double").write.parquet(str(health))
    completion_schema = "dataset string, battery_id string, cycle_index int, replay_sequence long, expected_telemetry_rows long, event_time timestamp, topic string, partition int, offset long"
    spark.createDataFrame([("MATR", "b", 1, 7, 2, datetime(2025, 1, 1, tzinfo=timezone.utc), "battery_lifecycle", 1, 7)],
        completion_schema).write.parquet(str(completed))
    spark.createDataFrame([("MATR", "b", 1, 1.1, 1.0, 0.01, 21.0, 25.0, 100.0)],
        "dataset string, battery_id string, cycle_index int, charge_capacity_in_Ah double, discharge_capacity_in_Ah double, internal_resistance_in_ohm double, temperature_min_in_C double, temperature_max_in_C double, charge_time_in_s double").write.parquet(str(canonical))
    spark.createDataFrame([("b", True, True)], "battery_id string, eol_observed boolean, replay_complete boolean").write.parquet(str(lifecycle))
    spark.createDataFrame([("b", "train", 1, "arrival"), ("validation", "validation", 2, "arrival")],
        "battery_id string, split string, arrival_rank int, schedule_fingerprint string").write.parquet(str(arrival))
    upsert_kafka_offset_watermarks(spark.createDataFrame([
        ("battery_measurements", 0, 9, "{}"), ("battery_lifecycle", 1, 7, "{}")
    ], "topic string, partition int, offset long, value string"), offsets)

    assert publish_completed_state(
        spark, health_path=health, completed_cycles_path=completed, offset_watermarks_path=offsets,
        state_root=tmp_path, canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        arrival_manifest_path=arrival, canonical_cycles_path=canonical, lifecycle_path=lifecycle, batch_id=0,
    ) is None
    assert not (tmp_path / "shared_feature_outlet").exists()
    spark.createDataFrame([("MATR", "b", 1, 7, 1, datetime(2025, 1, 1, tzinfo=timezone.utc), "battery_lifecycle", 1, 7)],
        completion_schema).write.mode("overwrite").parquet(str(completed))

    manifest = publish_completed_state(
        spark, health_path=health, completed_cycles_path=completed, offset_watermarks_path=offsets,
        state_root=tmp_path, canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        arrival_manifest_path=arrival, canonical_cycles_path=canonical, lifecycle_path=lifecycle, batch_id=1,
    )

    assert manifest["kafka_offsets"]["battery_lifecycle"]["1"] >= 7
    assert manifest["kafka_offsets"]["battery_measurements"]["0"] == 9
    assert manifest["arrival_manifest_fingerprint"] == "arrival"
    assert manifest["arrived_train_battery_ids"] == ["b"]
    assert manifest["observed_eol_train_battery_ids"] == ["b"]
    assert manifest["censored_train_battery_ids"] == []
    boundary = __import__("json").loads((tmp_path / manifest["finalized_cycle_boundary_ref"]).read_text())
    assert boundary["schema_version"] == "finalized-cycle-boundary-v2"
    assert boundary["finalized_cycle_ranges"][0]["finalized_cycle_count"] == 1
    outlet = spark.read.parquet(str(tmp_path / "shared_feature_outlet" / "segments"))
    segments = {path.name: path.read_bytes() for path in (tmp_path / "shared_feature_outlet" / "segments").glob("*.parquet")}
    assert outlet.count() == 1
    assert outlet.select("generation_id").first().generation_id == 2
    assert "soh" not in outlet.columns
    assert not (tmp_path / "historical_features").exists()
    cutoff = datetime.fromisoformat(manifest["cutoff_metadata"]["replay_cutoff"].replace("Z", "+00:00"))
    assert cutoff.astimezone(timezone.utc) == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert publish_completed_state(
        spark, health_path=health, completed_cycles_path=completed, offset_watermarks_path=offsets,
        state_root=tmp_path, canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        arrival_manifest_path=arrival, canonical_cycles_path=canonical, lifecycle_path=lifecycle, batch_id=2,
    ) is None
    assert {path.name: path.read_bytes() for path in (tmp_path / "shared_feature_outlet" / "segments").glob("*.parquet")} == segments
    assert spark.read.parquet(str(tmp_path / "shared_feature_outlet" / "segments")).count() == 1
    assert not (tmp_path / "historical_features").exists()
