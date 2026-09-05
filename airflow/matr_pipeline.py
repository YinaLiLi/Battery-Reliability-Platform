"""Canonical shared-state RUL/Survival retraining orchestration."""

from datetime import datetime, timedelta
import os

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


DEFAULT_ARGS = {
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
}


with DAG(
    dag_id="matr_shared_generation_retraining",
    description="Train RUL and Survival candidates from one immutable Streaming state and feature dataset.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    max_active_tasks=2,
    default_args=DEFAULT_ARGS,
    tags=["matr", "battery-reliability", "shared-generation"],
) as shared_generation_dag:
    validate_shared_inputs = BashOperator(
        task_id="validate_canonical_and_benchmark",
        bash_command=(
            "cd /opt/project && test -d data/processed/matr/cycle_summary && "
            "test -f data/processed/matr/shared_feature_outlet/_outlet.json && "
            "test -f data/processed/matr/fixed_offline_benchmark/v1/benchmark.json"
        ),
    )
    require_streaming_state = BashOperator(
        task_id="require_streaming_state",
        bash_command=(
            "cd /opt/project && state_manifest='{{ dag_run.conf.get(\"state_manifest\", \"\") }}' && "
            "if test -z \"$state_manifest\"; then echo >&2 "
            "'state_manifest is required; continuous training will not reconstruct historical state'; exit 2; fi && "
            "exec python3 src/shared_generation_receipt.py --state-manifest \"$state_manifest\" "
            "--generation {{ dag_run.conf.get('generation', '1.0') }} --validate-state-only --require-streaming"
        ),
        do_xcom_push=True,
    )
    finalize_receipt = BashOperator(
        task_id="finalize_shared_generation_receipt",
        bash_command=(
            "cd /opt/project && state_manifest='{{ ti.xcom_pull(task_ids=\"require_streaming_state\") }}' && "
            "exec python3 src/shared_generation_receipt.py --state-manifest \"$state_manifest\" "
            "--generation {{ dag_run.conf.get('generation', '1.0') }} --require-streaming"
        ),
        do_xcom_push=True,
    )
    train_rul = BashOperator(
        task_id="train_evaluate_rul",
        bash_command=(
            "cd /opt/project && receipt='{{ ti.xcom_pull(task_ids=\"finalize_shared_generation_receipt\") }}' && "
            "exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            "python3 src/train_matr_models.py --receipt \"$receipt\" --native-threads 1"
        ),
        pool="matr_native_training",
    )
    train_survival = DockerOperator(
        task_id="train_evaluate_survival",
        image="battery-reliability-survival-training:1.0",
        command=(
            "sh -c 'cd /opt/project && receipt=\"{{ ti.xcom_pull(task_ids=\"finalize_shared_generation_receipt\") }}\" && "
            "exec python src/survival_models.py --receipt \"$receipt\" --native-threads 1'"
        ),
        working_dir="/opt/project",
        mounts=[Mount(source=os.environ["HOST_PROJECT_ROOT"], target="/opt/project", type="bind")],
        environment={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
        network_mode="none",
        mem_limit="2g",
        cpus=1.0,
        auto_remove="success",
        mount_tmp_dir=False,
        pool="matr_native_training",
    )
    publish_rul = BashOperator(
        task_id="publish_rul_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping RUL publish for idempotent integration run'; else "
            "receipt='{{ ti.xcom_pull(task_ids=\"finalize_shared_generation_receipt\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt \"$receipt\" --artifact-dir rul) && "
            "exec python3 src/publish_predictions.py --artifact-dir \"$candidate_dir\"; fi"
        ),
    )
    load_rul = BashOperator(
        task_id="load_rul_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping RUL load for idempotent integration run'; else "
            "receipt='{{ ti.xcom_pull(task_ids=\"finalize_shared_generation_receipt\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt \"$receipt\" --artifact-dir rul) && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset battery_predictions --source-path \"$candidate_dir/published_predictions.parquet\" && "
            "exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset model_evaluations --source-path \"$candidate_dir/published_model_evaluation.parquet\"; fi"
        ),
    )
    publish_survival = BashOperator(
        task_id="publish_survival_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping Survival publish for idempotent integration run'; else "
            "receipt='{{ ti.xcom_pull(task_ids=\"finalize_shared_generation_receipt\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt \"$receipt\" --artifact-dir survival) && "
            "exec python3 src/publish_survival_predictions.py --artifact-dir \"$candidate_dir\"; fi"
        ),
    )
    load_survival = BashOperator(
        task_id="load_survival_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping Survival load for idempotent integration run'; else "
            "receipt='{{ ti.xcom_pull(task_ids=\"finalize_shared_generation_receipt\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt \"$receipt\" --artifact-dir survival) && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset battery_survival_predictions --source-path \"$candidate_dir/published_survival_predictions.parquet\" && "
            "exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset survival_model_evaluations --source-path \"$candidate_dir/published_survival_model_evaluation.parquet\"; fi"
        ),
    )

    validate_shared_inputs >> require_streaming_state >> finalize_receipt >> [train_rul, train_survival]
    train_rul >> publish_rul >> load_rul
    train_survival >> publish_survival >> load_survival
