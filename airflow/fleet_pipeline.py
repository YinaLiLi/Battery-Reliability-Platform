"""Manual local orchestration for the existing EV fleet batch pipeline."""

from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


DEFAULT_ARGS = {
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
}


with DAG(
    dag_id="fleet_batch_pipeline",
    description="Build labeled Spark features from the existing NASA-derived batch pipeline.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ev-fleet", "batch"],
) as dag:
    download_nasa_data = BashOperator(
        task_id="download_nasa_data",
        bash_command="cd /opt/project && exec python3 src/download_data.py",
        retries=2,
    )
    parse_nasa_cycles = BashOperator(
        task_id="parse_nasa_cycles",
        bash_command="cd /opt/project && exec python3 src/parse_nasa_data.py",
        retries=1,
    )
    simulate_fleet = BashOperator(
        task_id="simulate_fleet",
        bash_command="cd /opt/project && exec python3 src/fleet_simulator.py",
        retries=1,
    )
    build_spark_features = BashOperator(
        task_id="build_spark_features",
        bash_command=(
            "cd /opt/project && exec /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 src/spark_pipeline.py"
        ),
        retries=1,
    )

    download_nasa_data >> parse_nasa_cycles >> simulate_fleet >> build_spark_features
