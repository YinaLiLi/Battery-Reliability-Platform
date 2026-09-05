import shutil
import subprocess
import sys

import pytest

from src.feature_contract import RUL_FEATURES, aggregate_cycle_samples, feature_rows
from src.spark_pipeline import build_features


@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None or subprocess.run(["java", "-version"], capture_output=True).returncode:
        pytest.skip("Java is required for Spark tests")
    from pyspark.sql import SparkSession

    session = SparkSession.builder.master("local[1]").config("spark.pyspark.python", sys.executable).appName("matr-feature-test").getOrCreate()
    yield session
    session.stop()


def test_matr_features_use_current_and_prior_cycles_only(spark):
    cycles = spark.createDataFrame(
        [("MATR", "b1", 1, 1.0, 1.1, 20.0, 24.0, 100.0), ("MATR", "b1", 2, 0.9, 1.0, 21.0, 25.0, 110.0)],
        "dataset string, battery_id string, cycle_index int, discharge_capacity_in_Ah double, charge_capacity_in_Ah double, temperature_min_in_C double, temperature_max_in_C double, charge_time_in_s double",
    )
    provenance = spark.createDataFrame([("b1", "g1", "batch_1", "4C")], "battery_id string, lineage_group_id string, batch_id string, charge_policy string")
    measurements = spark.createDataFrame([("b1", 2, 3.1, -4.0), ("b1", 2, 3.3, -3.0)], "battery_id string, cycle_index int, voltage_in_V double, current_in_A double")

    rows = build_features(cycles, provenance, measurements).orderBy("cycle_index").collect()

    assert rows[0].prior_discharge_capacity_in_Ah is None
    assert rows[1].prior_discharge_capacity_in_Ah == 1.0
    assert rows[1].capacity_fade_from_prior == pytest.approx(-0.1)
    assert rows[1].voltage_mean_in_V == pytest.approx(3.2)


def test_historical_features_match_streaming_contract_for_finalized_cycles(spark):
    cycles = [
        ("MATR", "b1", cycle, 1.1 - cycle * 0.01, 1.2 - cycle * 0.01, 20.0 + cycle, 24.0 + cycle, 100.0 + cycle)
        for cycle in range(1, 12)
    ] + [("MATR", "b2", 1, None, 1.0, 20.0, 24.0, 100.0)]
    cycle_frame = spark.createDataFrame(cycles, "dataset string, battery_id string, cycle_index int, discharge_capacity_in_Ah double, charge_capacity_in_Ah double, temperature_min_in_C double, temperature_max_in_C double, charge_time_in_s double")
    provenance = spark.createDataFrame([("b1", "g1", "batch_1", "4C"), ("b2", "g2", "batch_1", "4C")], "battery_id string, lineage_group_id string, batch_id string, charge_policy string")
    samples = [
        {"dataset": "MATR", "battery_id": battery, "cycle_index": cycle, "sample_index": 0, "source_time_in_s": charge_time,
         "voltage_in_V": 3.0 + cycle / 100, "current_in_A": -4.0, "temperature_in_C": minimum,
         "charge_capacity_in_Ah": charge, "discharge_capacity_in_Ah": discharge, "internal_resistance_in_ohm": 0.01 + cycle / 1000,
         "replay_sequence": cycle}
        for _, battery, cycle, discharge, charge, minimum, _, charge_time in cycles
    ]
    samples.extend({**sample, "sample_index": 1, "source_time_in_s": sample["source_time_in_s"] + 1,
                    "voltage_in_V": sample["voltage_in_V"] + 0.1, "current_in_A": -3.0,
                    "temperature_in_C": sample["temperature_in_C"] + 4} for sample in list(samples))
    measurement_frame = spark.createDataFrame(samples)

    historical = {
        (row.battery_id, row.cycle_index): row.asDict()
        for row in build_features(cycle_frame, provenance, measurement_frame).collect()
    }
    by_cycle = {}
    for sample in samples:
        by_cycle.setdefault((sample["battery_id"], sample["cycle_index"]), []).append(sample)
    streaming = {(row["battery_id"], row["cycle_index"]): row for row in feature_rows(aggregate_cycle_samples(group) for group in by_cycle.values())}

    for key, stream_row in streaming.items():
        for feature in RUL_FEATURES:
            actual, expected = historical[key].get(feature), stream_row.get(feature)
            assert actual == expected if actual is None or expected is None else actual == pytest.approx(expected), (key, feature, actual, expected)
