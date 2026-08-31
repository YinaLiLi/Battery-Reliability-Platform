from pathlib import Path

from airflow.models import DagBag


ROOT = Path(__file__).resolve().parents[1]


def test_fleet_batch_pipeline_has_the_expected_linear_batch_tasks():
    dag_bag = DagBag(dag_folder=str(ROOT / "airflow"))

    assert not dag_bag.import_errors
    dag = dag_bag.dags.get("fleet_batch_pipeline")

    assert dag is not None
    assert dag.schedule is None
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert set(dag.task_ids) == {
        "download_nasa_data",
        "parse_nasa_cycles",
        "simulate_fleet",
        "build_spark_features",
    }
    assert dag.get_task("download_nasa_data").downstream_task_ids == {"parse_nasa_cycles"}
    assert dag.get_task("parse_nasa_cycles").downstream_task_ids == {"simulate_fleet"}
    assert dag.get_task("simulate_fleet").downstream_task_ids == {"build_spark_features"}
    assert not dag.get_task("build_spark_features").downstream_task_ids


def test_fleet_batch_pipeline_uses_the_defined_retry_and_spark_submission_policy():
    dag = DagBag(dag_folder=str(ROOT / "airflow")).dags.get("fleet_batch_pipeline")

    download = dag.get_task("download_nasa_data")
    spark = dag.get_task("build_spark_features")

    assert download.retries == 2
    assert spark.retries == 1
    assert download.retry_exponential_backoff is True
    assert spark.retry_exponential_backoff is True
    assert "spark://spark-master:7077" in spark.bash_command
    assert "/opt/spark/bin/spark-submit" in spark.bash_command
