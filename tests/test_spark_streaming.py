import shutil
import subprocess

import pytest

from src.spark_streaming import build_window_metrics, parse_telemetry


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None or subprocess.run(["java", "-version"], capture_output=True).returncode:
        pytest.skip("Java 17 is required to run local Spark tests")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.master("local[1]").appName("spark-streaming-test").getOrCreate()
    yield session
    session.stop()


def test_window_metrics_excludes_malformed_json_and_duplicate_event_ids(spark):
    kafka_records = spark.createDataFrame(
        [
            ('{"event_id":"one","vehicle_id":"EV-1","timestamp":"2025-01-01T00:00:00","pack_voltage":320.0,"module_temp_max":25.0}',),
            ('{"event_id":"one","vehicle_id":"EV-1","timestamp":"2025-01-01T00:00:00","pack_voltage":320.0,"module_temp_max":25.0}',),
            ('{"event_id":"two","vehicle_id":"EV-1","timestamp":"2025-01-01T01:00:00","pack_voltage":315.0,"module_temp_max":27.0}',),
            ("not-json",),
        ],
        "value string",
    )

    rows = build_window_metrics(parse_telemetry(kafka_records)).collect()

    assert len(rows) == 1
    assert rows[0].vehicle_id == "EV-1"
    assert rows[0].event_count == 2
    assert rows[0].average_pack_voltage == 317.5
    assert rows[0].maximum_module_temperature == 27.0
