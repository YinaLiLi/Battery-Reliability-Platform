"""Manual local orchestration for MATR reliability retraining."""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


DEFAULT_ARGS = {
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
}


with DAG(
    dag_id="matr_reliability_pipeline",
    description="Build measured SOH and RUL reliability artifacts for MATR.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["matr", "battery-reliability"],
) as dag:
    ingest_matr = BashOperator(
        task_id="ingest_matr",
        bash_command="cd /opt/project && test -f data/raw/batterylife/MATR.zip",
        retries=2,
    )
    normalize_matr = BashOperator(
        task_id="normalize_matr",
        bash_command="cd /opt/project && if test -f data/processed/matr/qc_report.json && test -d data/processed/matr/cycle_summary && test -d data/processed/matr/cycle_measurements; then echo 'Using validated MATR canonical outputs'; else exec python3 src/normalize_matr.py; fi",
        retries=1,
    )
    build_degradation_features = BashOperator(
        task_id="build_degradation_features",
        bash_command="cd /opt/project && if test -d data/processed/matr/degradation_features; then echo 'Using finalized MATR degradation features'; else exec /opt/spark/bin/spark-submit --master spark://spark-master:7077 src/spark_pipeline.py; fi",
        retries=1,
    )
    train_evaluate_models = BashOperator(
        task_id="train_evaluate_models",
        bash_command="cd /opt/project && exec python3 src/train_matr_models.py --model-version matr-rul-xgboost-{{ run_id | replace(':', '-') }}",
        retries=1,
    )
    publish_predictions = BashOperator(
        task_id="publish_predictions",
        bash_command="cd /opt/project && exec python3 src/publish_predictions.py",
        retries=1,
    )
    load_serving_tables = BashOperator(
        task_id="load_serving_tables",
        bash_command=(
            "cd /opt/project && /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset battery_cycle_health && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_predictions && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset battery_replay_windows && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 src/postgres_loader.py --dataset model_evaluations"
        ),
        retries=1,
    )

    ingest_matr >> normalize_matr >> build_degradation_features >> train_evaluate_models >> publish_predictions >> load_serving_tables


with DAG(
    dag_id="matr_continuous_retraining",
    description="Check daily simulated lifecycle eligibility and publish one RUL candidate generation.",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["matr", "battery-reliability", "continuous-retraining"],
) as continuous_dag:
    check_retraining_eligibility = BashOperator(
        task_id="check_retraining_eligibility",
        bash_command="cd /opt/project && exec python3 src/train_matr_models.py --continuous",
        retries=1,
    )
    load_candidate_generation = BashOperator(
        task_id="load_candidate_generation",
        bash_command=(
            "cd /opt/project && if test -f data/processed/matr/latest_candidate_generation.txt; then "
            "candidate_dir=$(cat data/processed/matr/latest_candidate_generation.txt) && "
            "python3 src/publish_predictions.py --artifact-dir $candidate_dir && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset battery_predictions --source-path $candidate_dir/published_predictions.parquet && "
            "/opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 "
            "src/postgres_loader.py --dataset model_evaluations --source-path $candidate_dir/published_model_evaluation.parquet; "
            "else echo 'No candidate generation to load'; fi"
        ),
        retries=1,
    )
    check_retraining_eligibility >> load_candidate_generation
