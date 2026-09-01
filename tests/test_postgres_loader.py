import shutil
import subprocess

import pytest

from src.postgres_loader import assert_group_totals_match, prepare_battery_cycle_health, prepare_evaluations, prepare_predictions, prepare_replay_windows, validate_snapshot


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None or subprocess.run(["java", "-version"], capture_output=True).returncode:
        pytest.skip("Java 17 is required to run local Spark tests")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.master("local[1]").appName("postgres-loader-test").getOrCreate()
    yield session
    session.stop()


def test_prepare_battery_cycle_health_uses_the_cycle_natural_key(spark):
    source = spark.createDataFrame(
        [("MATR", "MATR-1", 1, 0.98, 99, 1.05, 0.01, 28.0, 120.0, -0.001, 0.99)],
        "dataset string, battery_id string, cycle_index int, soh double, rul_cycles int, discharge_capacity_in_Ah double, internal_resistance_in_ohm double, temperature_max_in_C double, charge_time_in_s double, capacity_slope_10 double, coulombic_efficiency double",
    )
    row = prepare_battery_cycle_health(source).first()
    assert (row.dataset, row.battery_id, row.cycle_index, row.soh, row.rul_cycles) == ("MATR", "MATR-1", 1, 0.98, 99)


def test_prepare_replay_windows_uses_cycle_natural_key(spark):
    source = spark.createDataFrame(
        [("MATR-1", 1, 2, 3.2, 27.0, 1.0, 0.9, 0.01)],
        "battery_id string, cycle_index int, event_count long, average_voltage_in_V double, maximum_temperature_in_C double, charge_capacity_in_Ah double, discharge_capacity_in_Ah double, internal_resistance_in_ohm double",
    )
    row = prepare_replay_windows(source).first()
    assert (row.battery_id, row.cycle_index, row.event_count) == ("MATR-1", 1, 2)


def test_prepare_evaluations_defaults_missing_training_metadata_for_existing_artifacts(spark):
    source = spark.createDataFrame(
        [("candidate-1", "xgboost", "MATR", "candidate", "2026-09-01T00:00:00Z", "{}")],
        "model_version string, model_name string, dataset string, status string, evaluated_at string, metrics_json string",
    )

    row = prepare_evaluations(source).first()

    assert row.training_metadata == "{}"


def test_prediction_validation_rejects_negative_served_rul(spark):
    source = spark.createDataFrame(
        [("model-1", "MATR", "MATR-1", 1, -3.0, "2026-09-01T00:00:00Z", "test")],
        "model_version string, dataset string, battery_id string, cycle_index int, predicted_rul_cycles double, prediction_created_at string, split string",
    )

    with pytest.raises(ValueError, match="negative served RUL"):
        validate_snapshot(prepare_predictions(source), "battery_predictions")


def test_group_total_validation_rejects_a_mismatched_database_snapshot(spark):
    source = spark.createDataFrame([("south", 2), ("west", 3)], "region string, rows long")
    target = spark.createDataFrame([("south", 2), ("west", 2)], "region string, rows long")

    with pytest.raises(RuntimeError, match="group totals differ"):
        assert_group_totals_match(source, target, ("region",), "rows", "battery_cycle_health")
