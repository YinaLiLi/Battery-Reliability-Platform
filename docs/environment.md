# Run the platform

Choose your operating system and follow its Bash commands. The full workflow runs with native Docker on Linux, Docker Desktop on macOS, or Docker Desktop integrated with WSL2. Python **3.10–3.13**, Java **17+**, and the modern `docker compose` plugin are required.

## Linux

```bash
git clone <your-fork-or-release-checkout>
cd Battery-Reliability-Platform
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/raw/batterylife
cp .env.example .env
```

Set `HOST_PROJECT_ROOT` in `.env` to the absolute checkout path Docker can mount, for example:

```text
HOST_PROJECT_ROOT=/home/<user>/Battery-Reliability-Platform
```

Download the pinned public archives:

```bash
curl -fL 'https://zenodo.org/api/records/19688272/files/MATR.zip/content' -o data/raw/batterylife/MATR.zip
curl -fL 'https://zenodo.org/api/records/19688272/files/Life%20labels.zip/content' -o 'data/raw/batterylife/Life labels.zip'
md5sum data/raw/batterylife/MATR.zip 'data/raw/batterylife/Life labels.zip'
```

Expected MD5 values:

```text
83a1528858b9e1b7b6886757bb561669  data/raw/batterylife/MATR.zip
cd0cc01a7211972be45e8e38d86cdeca  data/raw/batterylife/Life labels.zip
```

## macOS

Use a Bash-compatible shell and Docker Desktop.

```bash
git clone <your-fork-or-release-checkout>
cd Battery-Reliability-Platform
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data/raw/batterylife
cp .env.example .env
```

Set `HOST_PROJECT_ROOT` in `.env`, for example:

```text
HOST_PROJECT_ROOT=/Users/<user>/.../Battery-Reliability-Platform
```

Download the Linux archive URLs above, then verify them with macOS `md5`:

```bash
md5 -q data/raw/batterylife/MATR.zip
md5 -q 'data/raw/batterylife/Life labels.zip'
```

Expected output:

```text
83a1528858b9e1b7b6886757bb561669
cd0cc01a7211972be45e8e38d86cdeca
```

On Apple Silicon, enable Docker Desktop amd64 emulation only when a required image, wheel, or native extension uses an amd64-only path. XGBoost/OpenMP requirements are provided by the pinned runtime images; use Docker Desktop rather than attempting to substitute host-native libraries.

## Windows (WSL2)

Run the complete platform inside a WSL2 Linux distribution with Docker Desktop WSL integration. Native PowerShell and `cmd` are not the full-stack execution path.

- Clone under the WSL Linux filesystem, for example `/home/<user>/Battery-Reliability-Platform`.
- Do not run the full workflow from `/mnt/c/...`.
- Run the Linux Bash commands in WSL2, including the `md5sum` checksum command.

Set `HOST_PROJECT_ROOT` in `.env` to the WSL path:

```text
HOST_PROJECT_ROOT=/home/<user>/Battery-Reliability-Platform
```

## Run the canonical workflow

After completing the applicable operating-system setup, first verify the environment:

```bash
python scripts/preflight.py --profile full --tracked
```

Then run the canonical progression from raw laboratory data to the dashboard:

```bash
python src/normalize_matr.py
python src/matr_qc.py
python scripts/preflight.py --profile full --data

docker compose --profile training build airflow dashboard postgres-init kafka topic-init spark-master spark-submit spark-stream-submit spark-postgres-load survival-serving survival-training
docker compose up -d postgres kafka spark-master spark-worker-1 spark-worker-2 airflow
docker compose run --rm postgres-init
docker compose run --rm topic-init
docker compose run --rm spark-submit
docker compose run --rm spark-stream-submit

python src/build_offline_benchmark.py
python src/kafka_producer.py
docker compose run --rm --no-deps spark-stream-submit

docker compose exec -T airflow airflow dags trigger matr_shared_generation_retraining --conf '{"generation":"1.3"}'
docker compose exec -T airflow airflow dags list-runs matr_shared_generation_retraining

docker compose run --rm spark-postgres-load /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 /opt/project/src/postgres_loader.py --initialize-current
docker compose run --rm spark-stream-submit /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 /opt/project/src/spark_streaming.py --refresh-current
docker compose run --rm survival-serving python -m src.survival_serving_worker --once

docker compose up -d dashboard
python scripts/preflight.py --profile compose --data
```

This sequence is: raw MATR → normalization/QC → Kafka → Spark Streaming → Shared Feature Outlet → benchmark → Airflow RUL/Survival → Current model initialization → serving refresh → PostgreSQL → Dashboard.
