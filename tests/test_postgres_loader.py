import shutil
import subprocess

import pytest

from src.postgres_loader import assert_group_totals_match, prepare_window_metrics


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None or subprocess.run(["java", "-version"], capture_output=True).returncode:
        pytest.skip("Java 17 is required to run local Spark tests")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.master("local[1]").appName("postgres-loader-test").getOrCreate()
    yield session
    session.stop()


def test_prepare_window_metrics_flattens_the_database_primary_key(spark):
    source = spark.createDataFrame(
        [("EV-1", "2025-01-01T00:00:00", "2025-01-01T06:00:00", 2, 317.5, 27.0)],
        "vehicle_id string, window_start string, window_end string, event_count long, average_pack_voltage double, maximum_module_temperature double",
    ).selectExpr(
        "vehicle_id",
        "named_struct('start', to_timestamp(window_start), 'end', to_timestamp(window_end)) AS window",
        "event_count",
        "average_pack_voltage",
        "maximum_module_temperature",
    )

    row = prepare_window_metrics(source).first()

    assert row.window_start.isoformat() == "2025-01-01T00:00:00"
    assert row.window_end.isoformat() == "2025-01-01T06:00:00"
    assert row.vehicle_id == "EV-1"


def test_group_total_validation_rejects_a_mismatched_database_snapshot(spark):
    source = spark.createDataFrame([("south", 2), ("west", 3)], "region string, rows long")
    target = spark.createDataFrame([("south", 2), ("west", 2)], "region string, rows long")

    with pytest.raises(RuntimeError, match="group totals differ"):
        assert_group_totals_match(source, target, ("region",), "rows", "vehicle_features")
