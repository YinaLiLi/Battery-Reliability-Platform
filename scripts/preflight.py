"""Repository environment checks; no third-party imports required to report failures."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_CAPABILITIES = (
    (('docker', 'compose', '--help'), ('--profile',)),
    (('docker', 'compose', 'config', '--help'), ('--quiet',)),
    (('docker', 'compose', 'run', '--help'), ('--rm', '--no-deps')),
    (('docker', 'compose', 'up', '--help'), ('--detach',)),
    (('docker', 'compose', 'down', '--help'), ('--volumes', '--remove-orphans')),
)


def python_error(version):
    if version < (3, 10):
        return 'Python >=3.10,<3.14 required: PySpark 4.1.3 requires Python >=3.10.'
    if version >= (3, 14):
        return 'Python >=3.10,<3.14 required: canonical Survival uses scikit-survival==0.24.1 and Python 3.10–3.13 wheels. Create a Python 3.12 virtual environment.'
    return None


def compose_version_supported(output):
    match = re.search(r'(?<!\d)v?(\d+)\.(\d+)(?:\.(\d+))?', output)
    return bool(match and tuple(map(int, match.groups(default='0'))) >= (2, 0, 0))


def missing_compose_capabilities(root, env):
    missing = []
    for command, flags in COMPOSE_CAPABILITIES:
        result = run(command, cwd=root, env=env)
        if result.returncode:
            missing.append(' '.join(command[2:]))
            continue
        output = result.stdout + result.stderr
        missing.extend(flag for flag in flags if flag not in output)
    return missing


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=120, **kwargs)


def environment(root):
    values = {}
    path = root / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('\"\'')
    return {**values, **os.environ}


def check(profile, root=ROOT, tracked=False, data=False):
    results = []
    def record(name, ok, message):
        results.append({'check': name, 'ok': bool(ok), 'message': message})
    error = python_error(sys.version_info[:2])
    record('Python', error is None, error or platform.python_version())
    if error:
        return results
    record('architecture', platform.machine().lower() in ('arm64', 'aarch64', 'amd64', 'x86_64'), platform.machine())
    native_windows = platform.system() == 'Windows'
    record('platform', not (native_windows and profile in ('compose', 'full')),
           'Full stack requires Linux containers; Windows: use Docker Desktop + WSL2 and clone inside the Linux filesystem.' if native_windows else platform.system())
    if tracked:
        tracked_files = set(run(['git', 'ls-files'], cwd=root).stdout.splitlines())
        required = {
            '.dockerignore', '.env.example', '.gitattributes', '.gitignore',
            '.github/workflows/ci.yml', 'pytest.ini',
            'airflow/matr_pipeline.py', 'dashboard/Dockerfile',
            'dashboard/app.py', 'dashboard/requirements.txt',
            'sql/001_analytics.sql', 'sql/002_drop_legacy_ev.sql',
            'sql/003_dashboard_role.sql',
        }
        required |= {p.relative_to(root).as_posix() for folder in ('src', 'scripts', 'tests') for p in (root / folder).glob('*.py')}
        required |= {p.name for p in root.glob('Dockerfile*')} | {p.name for p in root.glob('requirements*.txt')}
        for filename in ('README.md', 'docker-compose.yml'):
            contents = (root / filename).read_text(encoding='utf-8')
            required.update(re.findall(r'(?<![\w/])(?:src|scripts|docs|dashboard|tests)/[\w./-]+\.(?:py|md|txt)', contents))
            required.update(re.findall(r'dockerfile:\s*(\S+)', contents))
        missing = sorted(required - tracked_files)
        record('tracked files', not missing, ', '.join(missing) if missing else 'All referenced source/build files are tracked.')
    target = root / 'data' / 'processed'
    while not target.exists():
        target = target.parent
    record('data permissions', os.access(target, os.W_OK), 'Data parent must be writable: ' + str(target))
    # Container lockfiles reproduce serialized-model runtimes; host compatibility
    # is governed only by the developer requirements manifest.
    manifests = [] if profile == 'compose' else ['requirements.txt']
    for filename in manifests:
        for line in (root / filename).read_text().splitlines():
            match = re.fullmatch(r'([\w-]+)(?:\[.*?\])?([<>=!].*)', line)
            if not match:
                continue
            name, expected = match.groups()
            if profile in ('unit', 'spark') and name == 'scikit-survival':
                continue
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = 'missing'
            try:
                from packaging.specifiers import SpecifierSet
                compatible = actual != 'missing' and actual in SpecifierSet(expected)
            except ImportError:
                compatible = expected == '==' + actual
            record(name, compatible, f'{actual}; expected {expected} from {filename}; install with python -m pip install -r {filename}')
    if profile in ('spark', 'full'):
        worker = os.environ.get('PYSPARK_PYTHON', sys.executable)
        try:
            version = run([worker, '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'])
            record('Spark Python', version.returncode == 0 and version.stdout.strip() == '%s.%s' % sys.version_info[:2], 'PYSPARK_PYTHON must use the same Python minor version as the driver.')
        except (OSError, subprocess.TimeoutExpired):
            record('Spark Python', False, 'PYSPARK_PYTHON is not executable; point it to the project interpreter.')
        try:
            java = run(['java', '-version'])
            match = re.search(r'version "(\d+)', java.stderr + java.stdout)
            ok = java.returncode == 0 and match and int(match[1]) >= 17
            record('Java', ok, 'Java >=17 required; set JAVA_HOME to that JDK and add its bin directory to PATH.')
            home = os.environ.get('JAVA_HOME')
            home_ok = False
            if home:
                home_java = run([str(Path(home) / 'bin' / ('java.exe' if native_windows else 'java')), '-version'])
                home_version = re.search(r'version "(\d+)', home_java.stderr + home_java.stdout)
                home_ok = home_java.returncode == 0 and home_version and int(home_version[1]) >= 17
            record('JAVA_HOME', home_ok, 'Set JAVA_HOME to the Java >=17 installation.')
            spark_home = os.environ.get('SPARK_HOME')
            if spark_home:
                release = Path(spark_home) / 'RELEASE'
                record('Spark alignment', release.exists() and 'Spark 4.1.3' in release.read_text(), 'SPARK_HOME must contain Spark 4.1.3 matching PySpark and the Scala 2.13 Kafka connector; unset SPARK_HOME to use pip PySpark.')
        except (OSError, subprocess.TimeoutExpired):
            record('Java', False, 'Install Java >=17 and set JAVA_HOME.')
    if profile in ('survival', 'full'):
        try:
            imported = run([sys.executable, '-c', 'import sksurv, sklearn, numpy, joblib; from sksurv.ensemble import RandomSurvivalForest'])
            record('Survival imports', imported.returncode == 0, 'Pinned Survival native libraries must import; install requirements.survival-training.txt and requirements.survival-serving.txt.')
        except (OSError, subprocess.TimeoutExpired):
            record('Survival imports', False, 'Native Survival runtime could not start.')
    if profile in ('compose', 'full'):
        env = environment(root)
        for key in ('POSTGRES_PASSWORD', 'DASHBOARD_POSTGRES_PASSWORD', 'HOST_PROJECT_ROOT'):
            value = env.get(key, '')
            record(key, value and not value.lower().startswith('change-me'), f'Set {key} in .env; placeholder/empty values are unsupported.')
        record('host mount', env.get('HOST_PROJECT_ROOT') == str(root.resolve()), 'HOST_PROJECT_ROOT must equal the absolute checkout path in the Docker host filesystem.')
        try:
            version = run(['docker', 'compose', 'version', '--short'], cwd=root, env=env)
            supported = version.returncode == 0 and compose_version_supported(version.stdout + version.stderr)
            record('Compose CLI', supported, 'Compatible Docker Compose CLI found.' if supported else 'Docker Compose CLI >=2.0 required; legacy docker-compose v1 is unsupported.')
            missing = missing_compose_capabilities(root, env) if supported else ['Compose CLI unavailable']
            record('Compose capabilities', not missing, 'Required profile/config/run/up/down capabilities found.' if not missing else 'Missing required Compose capabilities: ' + ', '.join(missing))
        except (OSError, subprocess.TimeoutExpired):
            record('Compose CLI', False, 'Install a Docker Compose CLI >=2.0 compatible with the current Compose Specification.')
            record('Compose capabilities', False, 'Compose capability probes could not run.')
        for name, command in (
            ('Linux daemon', ['docker', 'info', '--format', '{{.OSType}}']),
            ('Compose config', ['docker', 'compose', 'config', '--quiet']),
            ('amd64 execution', ['docker', 'run', '--rm', '--platform', 'linux/amd64', 'python:3.10-slim', 'python', '-c', 'import platform; assert platform.machine() == "x86_64"']),
        ):
            try:
                result = run(command, cwd=root, env=env)
                ok = result.returncode == 0
                if name == 'Linux daemon':
                    ok = ok and result.stdout.strip() == 'linux'
                record(name, ok, 'Available' if ok else f'{name} unavailable; see docs/environment.md. Command output withheld to protect secrets.')
            except (OSError, subprocess.TimeoutExpired):
                record(name, False, 'Docker unavailable or timed out; start Docker and enable Linux containers/amd64 emulation.')
    if data:
        for path in ('data/processed/matr/cycle_summary', 'data/processed/matr/cycle_measurements', 'data/processed/matr/fixed_offline_benchmark/v1/benchmark.json'):
            record(path, (root / path).exists(), 'Run the data bootstrap in docs/environment.md.')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['unit', 'spark', 'survival', 'compose', 'full'], default='full')
    parser.add_argument('--tracked', action='store_true', help='Require git-tracked source/build references (CI).')
    parser.add_argument('--data', action='store_true', help='Require normalized data and benchmark before expensive jobs.')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    results = check(args.profile, tracked=args.tracked, data=args.data)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for row in results:
            print(f'{"PASS" if row["ok"] else "FAIL"} {row["check"]}: {row["message"]}')
    return int(any(not row['ok'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
