from pathlib import Path

import pytest

DagBag = pytest.importorskip("airflow.models").DagBag


def test_matr_pipeline_has_one_manifest_bound_retraining_path():
    dag = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "airflow")).dags.get("matr_retraining")
    assert dag is not None
    assert list(dag.task_ids) == ["validate_canonical_and_benchmark", "await_stream_state", "build_historical_features_as_of", "plan_candidate_generation", "train_evaluate_candidate", "publish_candidate", "load_candidate_serving"]
    assert dag.get_task("validate_canonical_and_benchmark").downstream_task_ids == {"await_stream_state"}
    assert dag.get_task("build_historical_features_as_of").downstream_task_ids == {"plan_candidate_generation"}
    assert dag.get_task("load_candidate_serving").downstream_task_ids == set()


def test_shared_generation_dag_branches_rul_and_survival_from_one_snapshot():
    dag = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "airflow")).dags.get("matr_shared_generation_retraining")
    assert dag is not None
    assert set(dag.task_ids) == {"validate_canonical_and_benchmark", "reconstruct_shared_snapshot", "build_historical_features_as_of", "train_evaluate_rul", "train_evaluate_survival", "publish_rul_candidate", "load_rul_candidate", "publish_survival_candidate", "load_survival_candidate"}
    assert dag.get_task("build_historical_features_as_of").downstream_task_ids == {"train_evaluate_rul", "train_evaluate_survival"}
    assert dag.get_task("train_evaluate_rul").downstream_task_ids == {"publish_rul_candidate"}
    assert dag.get_task("train_evaluate_survival").downstream_task_ids == {"publish_survival_candidate"}
    assert dag.get_task("publish_rul_candidate").downstream_task_ids == {"load_rul_candidate"}
    assert dag.get_task("publish_survival_candidate").downstream_task_ids == {"load_survival_candidate"}
    snapshot = dag.get_task("reconstruct_shared_snapshot")
    assert "generation_snapshots.py --generation" in snapshot.bash_command
    assert "shared_generation_state.txt" not in snapshot.bash_command
    assert dag.get_task("build_historical_features_as_of").pool == "matr_native_training"
    assert "--train-only" in dag.get_task("build_historical_features_as_of").bash_command
    assert dag.get_task("train_evaluate_rul").pool == "matr_native_training"
    assert dag.get_task("train_evaluate_survival").task_type == "DockerOperator"


def test_progressive_arrival_v3_dag_uses_the_schedule_bound_state_for_both_families():
    dag = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "airflow")).dags.get("matr_progressive_arrival_v3_retraining")
    assert dag is not None
    assert dag.get_task("build_historical_features_as_of").downstream_task_ids == {"train_evaluate_rul", "train_evaluate_survival"}
    assert "progressive_arrival.py --generation" in dag.get_task("reconstruct_progressive_snapshot").bash_command
    assert "progressive_arrival_v3/arrival_manifest.parquet" in dag.get_task("build_historical_features_as_of").bash_command
