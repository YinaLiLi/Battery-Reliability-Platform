import shutil
import subprocess
import sys

import pytest

from src.spark_streaming import build_lifecycle_state, build_window_metrics, parse_lifecycle, parse_telemetry, upsert_cycle_health


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
