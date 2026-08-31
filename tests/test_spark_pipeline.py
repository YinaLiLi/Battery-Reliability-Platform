import shutil
import subprocess
import sys
import os

import pytest

from src.spark_pipeline import build_features
from src.spark_pipeline import parse_args
from src.spark_pipeline import vehicle_dimension


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None or subprocess.run(["java", "-version"], capture_output=True).returncode:
        pytest.skip("Java 17 is required to run local Spark tests")
    from pyspark.sql import SparkSession

    master = os.environ.get("SPARK_TEST_MASTER", "local[1]")
    session = SparkSession.builder.master(master).appName("spark-pipeline-test").getOrCreate()
    yield session
    session.stop()


def test_build_features_adds_vehicle_window_features_and_labels(spark):
    telemetry = spark.createDataFrame(
        [
            ("one", "EV-1", "2025-01-01T00:00:00", "standard", "south", 320.0, 20.0, 25.0),
            ("two", "EV-1", "2025-01-01T01:00:00", "standard", "south", 315.0, 21.0, 27.0),
        ],
        "event_id string, vehicle_id string, timestamp string, battery_type string, region string, pack_voltage double, module_temp_min double, module_temp_max double",
    )
    labels = spark.createDataFrame(
        [
            ("one", "EV-1", "2025-01-01T00:00:00", 0),
            ("two", "EV-1", "2025-01-01T01:00:00", 1),
        ],
        "event_id string, vehicle_id string, timestamp string, failure_within_30_operating_days long",
    )

    features = build_features(telemetry, labels)
    rows = features.orderBy("event_id").collect()

    assert features.columns.count("vehicle_id") == 1
    assert features.columns.count("timestamp") == 1
    assert rows[0].module_temp_spread == 5.0
    assert rows[0].previous_pack_voltage is None
    assert rows[1].previous_pack_voltage == 320.0
    assert rows[1].rolling_module_temp_max == 27.0
    assert [row.failure_within_30_operating_days for row in rows] == [0, 1]
    assert {(row.vehicle_id, row.battery_type, row.region) for row in rows} == {("EV-1", "standard", "south")}


def test_broadcast_vehicle_dimension_matches_a_standard_join(spark):
    from pyspark.sql import functions as F

    telemetry = spark.createDataFrame(
        [("EV-1", "standard", "south"), ("EV-1", "standard", "south"), ("EV-2", "long_range", "west")],
        "vehicle_id string, battery_type string, region string",
    )
    dimension = vehicle_dimension(telemetry)

    standard = telemetry.join(dimension, "vehicle_id").orderBy("vehicle_id").collect()
    broadcast = telemetry.join(F.broadcast(dimension), "vehicle_id").orderBy("vehicle_id").collect()

    assert standard == broadcast


def test_parse_args_defaults_to_docker_spark_master(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spark_pipeline.py"])

    assert parse_args().master == "spark://spark-master:7077"
