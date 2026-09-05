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
python scripts/preflight.py --profile unit
python -m pytest --tier unit -q
cp .env.example .env
```

Native Windows unit tests, in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/preflight.py --profile unit
.venv\Scripts\python -m pytest --tier unit -q
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
docker compose up -d
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

Obtain the official BatteryLife MATR processed archive and life labels using the
download instructions in [BatteryLife](https://github.com/Ruifeng-Tan/BatteryLife).
Place them at `data/raw/batterylife/MATR.zip` and
`data/raw/batterylife/Life labels/MATR_labels.json`. They are not distributed by this
repository. Only use trusted publisher files: normalization loads Python pickle.
No sample results or ignored artifacts are prerequisites for the commands below.

```sh
python src/normalize_matr.py
docker compose run --rm spark-submit
python src/build_offline_benchmark.py
python scripts/preflight.py --profile compose --data
python src/kafka_producer.py --limit 100
docker compose run --rm spark-stream-submit
```

The bounded replay verifies infrastructure, not the full published experiment. The
normalizer creates the arrival manifest. Full replay (`python src/kafka_producer.py`)
is expensive and publishes the corpus. For continuous training, trigger
`matr_shared_generation_retraining` with `state_manifest` set to a Streaming-issued
finalized manifest and `generation` set to the intended generation. The DAG validates
Kafka lineage and the persistent shared feature outlet, finalizes one cumulative selection receipt,
and then fans out the same rows to RUL and Survival. Streaming appends only newly
prefix-complete cycles; no per-generation historical feature snapshot is created.
`generation_snapshots.py` remains the
explicit offline/backfill state-reconstruction entry point; direct trainer use from
that path requires `--offline-backfill`. Missing data/benchmark is caught by `--data`
before expensive jobs.
Measure disk and memory on the target machine before the full 130M-row experiment;
the bounded smoke is not a full-corpus resource certification.

## Verification and acceptance

```sh
python scripts/preflight.py --profile spark
python -m pytest --tier spark -q
python scripts/preflight.py --profile survival
python -m pytest --tier survival -q
docker compose run --rm --no-deps airflow python3 -m pytest --tier airflow -q
python -m pytest --tier integration -q
```

CI runs fresh checkouts for Python 3.10–3.13 and reports all skipped tests as failures.
`--tracked` checks source/build references against `git ls-files`; developers must
include new canonical files in their reviewed commit before a fork can use them.
The Survival roundtrip test proves a newly trained model reloads in the pinned
runtime; it does not authorize changing versions for existing serialized models.

Manual release gates: repeat these commands on current macOS Docker Desktop and
Windows WSL2, including a directory containing spaces, and record OS, architecture,
Python, Java, Docker/Compose versions and outcomes in the release evidence. Until
then these paths are targets with an explicit testing gap, not verified claims.
Test missing secrets, unsupported Python, missing Java, and Docker stopped; expect
an actionable nonzero preflight result. Use fresh volumes for acceptance, never
someone's existing data. Stop with `docker compose down`; for the disposable test
stack only, `docker compose down --volumes --remove-orphans` deletes its databases.
