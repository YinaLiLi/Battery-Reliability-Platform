from pathlib import Path

import pytest

DagBag = pytest.importorskip("airflow.models").DagBag


def test_only_canonical_shared_generation_dag_is_discoverable():
    dags = DagBag(dag_folder=str(Path(__file__).resolve().parents[1] / "airflow")).dags
    assert set(dags) == {"matr_shared_generation_retraining"}
    dag = dags["matr_shared_generation_retraining"]
    assert dag is not None
    assert set(dag.task_ids) == {"validate_canonical_and_benchmark", "require_streaming_state", "finalize_shared_generation_receipt", "train_evaluate_rul", "train_evaluate_survival", "publish_rul_candidate", "load_rul_candidate", "publish_survival_candidate", "load_survival_candidate"}
    assert dag.get_task("finalize_shared_generation_receipt").downstream_task_ids == {"train_evaluate_rul", "train_evaluate_survival"}
    assert dag.get_task("train_evaluate_rul").downstream_task_ids == {"publish_rul_candidate"}
    assert dag.get_task("train_evaluate_survival").downstream_task_ids == {"publish_survival_candidate"}
    assert dag.get_task("publish_rul_candidate").downstream_task_ids == {"load_rul_candidate"}
    assert dag.get_task("publish_survival_candidate").downstream_task_ids == {"load_survival_candidate"}
    state = dag.get_task("require_streaming_state")
    assert "generation_snapshots.py" not in state.bash_command
    assert "--require-streaming" in state.bash_command
    assert "cycle_measurements" not in dag.get_task("validate_canonical_and_benchmark").bash_command
    assert "shared_feature_outlet" in dag.get_task("validate_canonical_and_benchmark").bash_command
    assert dag.get_task("train_evaluate_rul").pool == "matr_native_training"
    assert dag.get_task("train_evaluate_survival").task_type == "DockerOperator"
    assert "--receipt" in dag.get_task("train_evaluate_rul").bash_command
    assert "--receipt" in dag.get_task("train_evaluate_survival").command
