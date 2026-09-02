from pathlib import Path

import pytest

DagBag = pytest.importorskip("airflow.models").DagBag


def test_matr_pipeline_has_one_linear_retraining_path():
    dag = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "airflow")).dags.get("matr_reliability_pipeline")
    assert dag is not None
    assert list(dag.task_ids) == ["ingest_matr", "normalize_matr", "build_degradation_features", "train_evaluate_models", "publish_predictions", "load_serving_tables"]
    assert dag.get_task("ingest_matr").downstream_task_ids == {"normalize_matr"}
    assert dag.get_task("train_evaluate_models").downstream_task_ids == {"publish_predictions"}
    assert dag.get_task("load_serving_tables").downstream_task_ids == set()
    assert "exec /opt/spark/bin/spark-submit" not in dag.get_task("load_serving_tables").bash_command


def test_continuous_retraining_dag_checks_daily_before_training_a_candidate():
    dag = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "airflow")).dags.get("matr_continuous_retraining")
    assert dag is not None
    assert dag.schedule is not None
    assert list(dag.task_ids) == ["check_retraining_eligibility", "load_candidate_generation"]
    assert "--continuous" in dag.get_task("check_retraining_eligibility").bash_command
    loader = dag.get_task("load_candidate_generation").bash_command
    assert "publish_predictions.py --artifact-dir $candidate_dir" in loader
    assert "published_predictions.parquet" in loader
