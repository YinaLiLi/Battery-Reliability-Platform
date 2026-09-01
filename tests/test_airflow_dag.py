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
