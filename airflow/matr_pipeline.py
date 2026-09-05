"""Manual local orchestration for MATR reliability retraining."""

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
    dag_id="matr_retraining",
    description="Build manifest-bound historical features and conditional RUL candidates.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["matr", "battery-reliability"],
) as dag:
    validate_canonical_and_benchmark = BashOperator(
        task_id="validate_canonical_and_benchmark",
        bash_command="cd /opt/project && test -d data/processed/matr/cycle_summary && test -d data/processed/matr/cycle_measurements && test -f data/processed/matr/fixed_offline_benchmark/v1/benchmark.json",
    )
    await_stream_state = BashOperator(
        task_id="await_stream_state",
        bash_command="cd /opt/project && test -f data/processed/matr/stream_state/latest.json",
    )
    build_historical_features_as_of = BashOperator(
        task_id="build_historical_features_as_of",
        bash_command="cd /opt/project && state_id=$(python3 -c \"import json; print(json.load(open('data/processed/matr/stream_state/latest.json'))['state_id'])\") && exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 src/spark_pipeline.py --state-manifest data/processed/matr/stream_state/latest.json --finalized-cycle-boundary data/processed/matr/finalized_cycle_boundary/$state_id/boundary.json --train-only",
    )
    plan_candidate_generation = BashOperator(
        task_id="plan_candidate_generation",
        bash_command="cd /opt/project && exec python3 src/train_matr_models.py --continuous",
    )
    train_evaluate_candidate = BashOperator(
        task_id="train_evaluate_candidate",
        bash_command="cd /opt/project && test -f data/processed/matr/latest_candidate_generation.txt || echo 'No candidate generation due'",
    )
    publish_candidate = BashOperator(
        task_id="publish_candidate",
        bash_command="cd /opt/project && if test -f data/processed/matr/latest_candidate_generation.txt; then candidate_dir=$(cat data/processed/matr/latest_candidate_generation.txt) && exec python3 src/publish_predictions.py --artifact-dir $candidate_dir; else echo 'No candidate generation due'; fi",
    )
    load_candidate_serving = BashOperator(
        task_id="load_candidate_serving",
        bash_command=(
            "cd /opt/project && if test -f data/processed/matr/latest_candidate_generation.txt; then candidate_dir=$(cat data/processed/matr/latest_candidate_generation.txt) && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_predictions --source-path $candidate_dir/published_predictions.parquet && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset model_evaluations --source-path $candidate_dir/published_model_evaluation.parquet; else echo 'No candidate generation due'; fi"
        ),
    )

    validate_canonical_and_benchmark >> await_stream_state >> build_historical_features_as_of >> plan_candidate_generation >> train_evaluate_candidate >> publish_candidate >> load_candidate_serving


with DAG(
    dag_id="matr_shared_generation_retraining",
    description="Train RUL and survival candidates from one immutable Streaming state.",
    start_date=datetime(2025, 1, 1), schedule=None, catchup=False, max_active_runs=1, max_active_tasks=2,
    default_args=DEFAULT_ARGS, tags=["matr", "battery-reliability", "shared-generation"],
) as shared_generation_dag:
    validate_shared_inputs = BashOperator(
        task_id="validate_canonical_and_benchmark",
        bash_command="cd /opt/project && test -d data/processed/matr/cycle_summary && test -f data/processed/matr/fixed_offline_benchmark/v1/benchmark.json",
    )
    reconstruct_shared_snapshot = BashOperator(
        task_id="reconstruct_shared_snapshot",
        bash_command=(
            "cd /opt/project && if test -n '{{ dag_run.conf.get('receipt', '') }}'; then "
            "test -f '{{ dag_run.conf.get('receipt', '') }}' && printf '%s\\n' '{{ dag_run.conf.get('receipt', '') }}'; "
            "else state_manifest=$(python3 src/generation_snapshots.py --generation "
            "{{ dag_run.conf.get('generation', '1.0') }}) && exec python3 src/shared_generation_receipt.py "
            "--state-manifest $state_manifest --generation {{ dag_run.conf.get('generation', '1.0') }}; fi"
        ),
        do_xcom_push=True,
    )
    build_shared_features = BashOperator(
        task_id="build_historical_features_as_of",
        bash_command=(
            "cd /opt/project && receipt='{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}' && "
            "state_manifest=$(python3 src/shared_generation_receipt.py --receipt $receipt --field state_manifest_path) && "
            "state_id=$(python3 src/shared_generation_receipt.py --receipt $receipt --field state_id) && "
            "exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 src/spark_pipeline.py "
            "--state-manifest $state_manifest --finalized-cycle-boundary data/processed/matr/finalized_cycle_boundary/$state_id/boundary.json --train-only"
        ),
        pool="matr_native_training",
    )
    train_rul_from_shared_state = BashOperator(
        task_id="train_evaluate_rul",
        bash_command=(
            "cd /opt/project && receipt='{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}' && "
            "state_manifest=$(python3 src/shared_generation_receipt.py --receipt $receipt --field state_manifest_path) && "
            "features=$(python3 src/shared_generation_receipt.py --receipt $receipt --field training_features_path) && "
            "exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            "python3 src/train_matr_models.py --state-manifest $state_manifest --generation {{ dag_run.conf.get('generation', '1.0') }} "
            "--training-features $features --native-threads 1"
        ),
        pool="matr_native_training",
    )
    train_survival_from_shared_state = DockerOperator(
        task_id="train_evaluate_survival",
        image="battery-reliability-survival-training:1.0",
        command=(
            "sh -c 'cd /opt/project && receipt=\"{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}\" && "
            "state_manifest=$(python src/shared_generation_receipt.py --receipt $receipt --field state_manifest_path) && "
            "features=$(python src/shared_generation_receipt.py --receipt $receipt --field training_features_path) && "
            "exec python src/survival_models.py --state-manifest $state_manifest --generation {{ dag_run.conf.get('generation', '1.0') }} "
            "--training-features $features --native-threads 1'"
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
    publish_rul_candidate = BashOperator(
        task_id="publish_rul_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping RUL publish for idempotent integration run'; else receipt='{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt $receipt --artifact-dir rul) && "
            "exec python3 src/publish_predictions.py --artifact-dir $candidate_dir; fi"
        ),
    )
    load_rul_candidate = BashOperator(
        task_id="load_rul_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping RUL load for idempotent integration run'; else receipt='{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt $receipt --artifact-dir rul) && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_predictions --source-path $candidate_dir/published_predictions.parquet && "
            "exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset model_evaluations --source-path $candidate_dir/published_model_evaluation.parquet; fi"
        ),
    )
    publish_survival_candidate = BashOperator(
        task_id="publish_survival_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping Survival publish for idempotent integration run'; else receipt='{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt $receipt --artifact-dir survival) && "
            "exec python3 src/publish_survival_predictions.py --artifact-dir $candidate_dir; fi"
        ),
    )
    load_survival_candidate = BashOperator(
        task_id="load_survival_candidate",
        bash_command=(
            "cd /opt/project && if test '{{ dag_run.conf.get('skip_publish_load', false) }}' = 'true'; then "
            "echo 'Skipping Survival load for idempotent integration run'; else receipt='{{ ti.xcom_pull(task_ids=\"reconstruct_shared_snapshot\") }}' && "
            "candidate_dir=$(python3 src/shared_generation_receipt.py --receipt $receipt --artifact-dir survival) && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_survival_predictions --source-path $candidate_dir/published_survival_predictions.parquet && "
            "exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset survival_model_evaluations --source-path $candidate_dir/published_survival_model_evaluation.parquet; fi"
        ),
    )
    validate_shared_inputs >> reconstruct_shared_snapshot >> build_shared_features >> [train_rul_from_shared_state, train_survival_from_shared_state]
    train_rul_from_shared_state >> publish_rul_candidate >> load_rul_candidate
    train_survival_from_shared_state >> publish_survival_candidate >> load_survival_candidate


with DAG(
    dag_id="matr_progressive_arrival_v3_retraining",
    description="Train RUL and survival candidates from one natural progressive-arrival state.",
    start_date=datetime(2025, 1, 1), schedule=None, catchup=False, max_active_runs=1,
    default_args=DEFAULT_ARGS, tags=["matr", "battery-reliability", "progressive-arrival-v3"],
) as progressive_arrival_v3_dag:
    validate_v3_inputs = BashOperator(
        task_id="validate_canonical_and_benchmark",
        bash_command="cd /opt/project && test -d data/processed/matr/cycle_summary && test -f data/processed/matr/fixed_offline_benchmark/v1/benchmark.json",
    )
    reconstruct_v3_snapshot = BashOperator(
        task_id="reconstruct_progressive_snapshot",
        bash_command="cd /opt/project && exec python3 src/progressive_arrival.py --generation {{ dag_run.conf.get('generation', '1.0') }} --latest data/processed/matr/progressive_v3_state.txt",
    )
    build_v3_features = BashOperator(
        task_id="build_historical_features_as_of",
        bash_command="cd /opt/project && state_manifest=$(cat data/processed/matr/progressive_v3_state.txt) && state_id=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['state_id'])\" $state_manifest) && exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 src/spark_pipeline.py --state-manifest $state_manifest --finalized-cycle-boundary data/processed/matr/finalized_cycle_boundary/$state_id/boundary.json --arrival-manifest data/processed/matr/progressive_arrival_v3/arrival_manifest.parquet --train-only",
    )
    train_v3_rul = BashOperator(
        task_id="train_evaluate_rul",
        bash_command="cd /opt/project && state_manifest=$(cat data/processed/matr/progressive_v3_state.txt) && state_id=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['state_id'])\" $state_manifest) && exec python3 src/train_matr_models.py --state-manifest $state_manifest --generation {{ dag_run.conf.get('generation', '1.0') }} --training-features data/processed/matr/historical_features/$state_id",
    )
    train_v3_survival = BashOperator(
        task_id="train_evaluate_survival",
        bash_command="cd /opt/project && state_manifest=$(cat data/processed/matr/progressive_v3_state.txt) && state_id=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['state_id'])\" $state_manifest) && exec python3 src/survival_models.py --state-manifest $state_manifest --generation {{ dag_run.conf.get('generation', '1.0') }} --training-features data/processed/matr/historical_features/$state_id",
    )
    publish_v3_rul = BashOperator(
        task_id="publish_load_rul_candidate",
        bash_command="cd /opt/project && candidate_dir=$(cat data/processed/matr/latest_candidate_generation.txt) && python3 src/publish_predictions.py --artifact-dir $candidate_dir && /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_predictions --source-path $candidate_dir/published_predictions.parquet && exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset model_evaluations --source-path $candidate_dir/published_model_evaluation.parquet",
    )
    publish_v3_survival = BashOperator(
        task_id="publish_load_survival_candidate",
        bash_command="cd /opt/project && candidate_dir=$(cat data/processed/matr/latest_survival_candidate_generation.txt) && python3 src/publish_survival_predictions.py --artifact-dir $candidate_dir && /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_survival_predictions --source-path $candidate_dir/published_survival_predictions.parquet && exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset survival_model_evaluations --source-path $candidate_dir/published_survival_model_evaluation.parquet",
    )
    validate_v3_inputs >> reconstruct_v3_snapshot >> build_v3_features >> [train_v3_rul, train_v3_survival]
    train_v3_rul >> publish_v3_rul
    train_v3_survival >> publish_v3_survival
