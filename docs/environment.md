# Supported environment and clean-clone setup

The integrated project targets Python **>=3.10,<3.14**, Java **>=17**, Docker
Compose CLI **>=2.0 with the required Compose Specification capabilities**, and
Linux containers. All preflight profiles enforce the same
Python range. Python 3.14 is outside project support because canonical Survival
uses scikit-survival 0.24.1 and its Python 3.10–3.13 wheel set.
The requirements files preserve model/runtime dependency versions; they are not
requirements to match a developer's OS or interpreter patch version.

Linux x86-64 is the full-stack CI target. macOS uses Docker Desktop; Windows uses
Docker Desktop with WSL2 integration and a Linux-filesystem checkout (not `/mnt/c`).
These desktop full-stack paths remain unverified until their manual acceptance
runs are recorded. Native Windows runs unit tests only. Linux ARM Survival uses
amd64 emulation; macOS ARM wheels do not imply Linux ARM wheel availability.
XGBoost on macOS requires an OpenMP runtime (`brew install libomp`); Windows may
require the Microsoft Visual C++ runtime. Import errors must be resolved before
training. The dashboard, Kafka 4.1.0 and PostgreSQL 16 run in containers.

## Clone and Python environment

Clone your fork and enter its root. On Linux, macOS, or WSL2:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Native Windows unit tests, in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/preflight.py --profile unit
```

Any Python 3.10–3.13 may replace 3.12. No subsystem needs a different host Python.
For local Spark set JAVA_HOME to your JDK >=17 and put its bin directory on PATH.
Dockerized Spark uses its own Java; host Java is checked by `spark`/`full`.

## Configure and start containers

Edit `.env`: replace both password placeholders and set HOST_PROJECT_ROOT to the
absolute checkout directory, as seen by the local Docker daemon. Do not use a
remote Docker context: Airflow launches sibling containers with that bind mount.
The Docker socket gives Airflow control of the local daemon; use this stack on a
trusted development machine. No secrets belong in git. Compose reads `.env`;
ordinary Python commands do not automatically export it.

```sh
python scripts/preflight.py --profile compose
docker compose config --quiet
docker compose build spark-master
docker compose --profile training build airflow dashboard survival-serving survival-training
docker compose up -d postgres kafka spark-master spark-worker-1 spark-worker-2 airflow
```

Preflight's amd64 check runs a disposable Python container and may pull its image.
Enable amd64 emulation on ARM machines if this fails. Legacy Compose v1 is
unsupported. Newer Compose releases remain supported when the CLI provides profiles,
configuration validation, one-shot runs, detached startup, and disposable teardown.
One-shot Spark jobs are in the `jobs` profile so startup does not run jobs before data exists.
Survival training is built explicitly before Airflow needs its image.

URLs: dashboard http://localhost:8501, Airflow http://localhost:8083, Spark master
http://localhost:8080. PostgreSQL uses localhost:5432 from the host; Kafka uses
localhost:9092. Inside Compose use postgres:5432 and kafka:29092. Airflow standalone
credentials are reported by Airflow; consult `docker compose logs airflow` locally.

## Data bootstrap

Use the pinned official BatteryLife processed-data release at Zenodo record
`19688272`. Download exactly these two files; they are not distributed here:

| File | Bytes | MD5 |
|---|---:|---|
| `MATR.zip` | 4,864,920,138 | `83a1528858b9e1b7b6886757bb561669` |
| `Life labels.zip` | 12,586 | `cd0cc01a7211972be45e8e38d86cdeca` |

```sh
mkdir -p data/raw/batterylife
curl -fL 'https://zenodo.org/api/records/19688272/files/MATR.zip/content' -o data/raw/batterylife/MATR.zip
curl -fL 'https://zenodo.org/api/records/19688272/files/Life%20labels.zip/content' -o 'data/raw/batterylife/Life labels.zip'
md5sum data/raw/batterylife/MATR.zip 'data/raw/batterylife/Life labels.zip'
python -c "from pathlib import Path; from src.normalize_matr import extract_archive; extract_archive(Path('data/raw/batterylife/Life labels.zip'))"
test -f 'data/raw/batterylife/Life labels/MATR_labels.json'
```

On macOS, replace `md5sum FILE` with `md5 -q FILE`. Verify both hashes before
normalization: the archive contains trusted publisher pickle data. Budget at least
100 GiB of free disk and 16 GiB RAM for the complete 130-million-measurement run.
The default Spark cluster uses two 1-core, 1 GiB workers; runtime varies
substantially with CPU, storage, and Docker memory limits.

Normalize and validate before starting event replay:

```sh
python src/normalize_matr.py
python src/matr_qc.py
python scripts/preflight.py --profile full
docker compose config --quiet
docker compose build spark-master
docker compose --profile training build airflow dashboard survival-serving survival-training
docker compose up -d postgres kafka spark-master spark-worker-1 spark-worker-2 airflow
docker compose run --rm postgres-init
docker compose run --rm topic-init
docker compose run --rm --no-deps spark-submit
```

The canonical workflow has one producer of finalized state and shared features:
Spark Streaming. Offline state reconstruction is not part of the public workflow.

```sh
python src/kafka_producer.py
docker compose run --rm spark-stream-submit
test -f data/processed/matr/stream_state/latest.json
test -f data/processed/matr/shared_feature_outlet/_outlet.json
python src/build_offline_benchmark.py
python scripts/preflight.py --profile compose --data
```

Streaming available-now processing exits after the current Kafka offsets are
finalized. `SPARK_MAX_OFFSETS_PER_TRIGGER` defaults to 5,000,000 for the full
corpus; lower it only if a worker cannot process that batch size.

Trigger the shared Airflow generation after substituting the path printed by the
first command:

```sh
STATE_MANIFEST=$(python -c "import json; from pathlib import Path; p=Path('data/processed/matr/stream_state/latest.json'); m=json.loads(p.read_text()); print(Path('data/processed/matr') / 'stream_state' / m['state_id'] / 'manifest.json')")
docker compose exec -T airflow airflow dags unpause matr_shared_generation_retraining
docker compose exec -T airflow airflow dags trigger matr_shared_generation_retraining --conf "{\"state_manifest\":\"$STATE_MANIFEST\",\"generation\":\"1.3\"}"
docker compose exec -T airflow airflow dags list-runs matr_shared_generation_retraining
```

Wait for the run to succeed and for both candidate evaluation tables to be loaded.
Then initialize both Current selections once, and explicitly refresh serving without
publishing any additional Kafka events:

```sh
docker compose run --rm spark-postgres-load /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 /opt/project/src/postgres_loader.py --initialize-current
docker compose run --rm spark-stream-submit /opt/spark/bin/spark-submit --master spark://spark-master:7077 --packages org.postgresql:postgresql:42.7.7 /opt/project/src/spark_streaming.py --refresh-current
docker compose run --rm survival-serving python -m src.survival_serving_worker --once
docker compose run --rm spark-postgres-load
docker compose up -d survival-serving dashboard
```

The initializer uses stable `evaluated_at`, then `model_version`, ordering and does
not replace an existing Current selection. The Dashboard remains available at
http://localhost:8501 after PostgreSQL is populated.

## Verification and acceptance

Run `python scripts/preflight.py --profile full --tracked --data`. The tracked-file
guard requires `git ls-files` to match `.public-files` exactly. Public CI checks this
product surface, container builds, Spark smoke, and SQL/bootstrap behavior.

Manual release gates: repeat these commands on current macOS Docker Desktop and
Windows WSL2, including a directory containing spaces, and record OS, architecture,
Python, Java, Docker/Compose versions and outcomes in the release evidence. Until
then these paths are targets with an explicit testing gap, not verified claims.
Test missing secrets, unsupported Python, missing Java, and Docker stopped; expect
an actionable nonzero preflight result. Use fresh volumes for acceptance, never
someone's existing data. Stop with `docker compose down`; for the disposable test
stack only, `docker compose down --volumes --remove-orphans` deletes its databases.
