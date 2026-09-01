import shutil
import subprocess
import sys

import pytest

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
