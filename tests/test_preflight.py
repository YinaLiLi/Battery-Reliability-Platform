import subprocess
import sys
from pathlib import Path
import zipfile
import pytest

from scripts import preflight
from src.normalize_matr import extract_archive


def test_unified_python_range():
    assert 'PySpark' in preflight.python_error((3, 9))
    assert 'Survival' in preflight.python_error((3, 14))
    assert all(preflight.python_error((3, minor)) is None for minor in range(10, 14))


def test_missing_services_and_secrets_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(preflight, 'environment', lambda root: {})
    def missing(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(preflight, 'run', missing)
    rows = {row['check']: row for row in preflight.check('compose', tmp_path)}
    for key in ('platform', 'POSTGRES_PASSWORD', 'Compose CLI', 'Linux daemon', 'amd64 execution'):
        assert not rows[key]['ok']
    assert 'WSL2' in rows['platform']['message']


def test_missing_java(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('')
    def missing(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(preflight, 'run', missing)
    rows = preflight.check('spark', tmp_path)
    assert any(row['check'] == 'Java' and not row['ok'] for row in rows)


def test_archive_spaces_and_traversal(tmp_path):
    archive = tmp_path / 'MATR data.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('MATR data/cell.pkl', b'test')
    extract_archive(archive)
    assert (tmp_path / 'MATR data/cell.pkl').read_bytes() == b'test'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('../escape', b'bad')
    import pytest
    with pytest.raises(ValueError, match='outside'):
        extract_archive(archive)


def test_endpoint_environment(monkeypatch):
    from src import kafka_consumer, kafka_producer
    monkeypatch.setattr(sys, 'argv', ['command'])
    monkeypatch.setenv('KAFKA_BOOTSTRAP_SERVERS', 'broker:29092')
    assert kafka_consumer.parse_args().bootstrap_server == 'broker:29092'
    assert kafka_producer.parse_args().bootstrap_server == 'broker:29092'


@pytest.mark.parametrize('version', [(3, 9), (3, 14)])
def test_unsupported_python_stops_before_external_commands(tmp_path, monkeypatch, version):
    monkeypatch.setattr(preflight.sys, 'version_info', version)
    monkeypatch.setattr(preflight, 'run', lambda *a, **k: pytest.fail('Must fail before executing anything'))
    for profile in ('unit', 'spark', 'survival', 'compose', 'full'):
        rows = preflight.check(profile, tmp_path)
        assert len(rows) == 1 and not rows[0]['ok']
        assert 'Python >=3.10,<3.14' in rows[0]['message']


@pytest.mark.parametrize('failure', ['Compose v1', 'Compose missing', 'capability', 'daemon', 'amd64', 'secret missing', 'secret placeholder'])
def test_stack_failure_matrix(tmp_path, monkeypatch, failure):
    env = dict(POSTGRES_PASSWORD='test-password', DASHBOARD_POSTGRES_PASSWORD='test-password', HOST_PROJECT_ROOT=str(tmp_path))
    if failure == 'secret missing':
        env.pop('POSTGRES_PASSWORD')
    if failure == 'secret placeholder':
        env['POSTGRES_PASSWORD'] = 'change-me'
    monkeypatch.setattr(preflight, 'environment', lambda root: env)
    monkeypatch.setattr(preflight.platform, 'system', lambda: 'Linux')
    def fake_run(args, **kwargs):
        output, code = '', 0
        if args[1:3] == ['compose', 'version']:
            if failure == 'Compose missing':
                raise FileNotFoundError()
            output = '1.29.2' if failure == 'Compose v1' else '2.39.0'
        elif '--help' in args:
            output = '' if failure == 'capability' else '--profile --no-deps --rm --detach --volumes --remove-orphans'
        elif args[1] == 'info':
            output, code = 'linux', int(failure == 'daemon')
        elif args[1] == 'run':
            code = int(failure == 'amd64')
        return subprocess.CompletedProcess(args, code, output, '')
    monkeypatch.setattr(preflight, 'run', fake_run)
    rows = {row['check']: row for row in preflight.check('compose', tmp_path)}
    name = {'Compose v1': 'Compose CLI', 'Compose missing': 'Compose CLI', 'capability': 'Compose capabilities', 'daemon': 'Linux daemon', 'amd64': 'amd64 execution', 'secret missing': 'POSTGRES_PASSWORD', 'secret placeholder': 'POSTGRES_PASSWORD'}[failure]
    assert not rows[name]['ok']
    assert rows[name]['message']
    assert 'test-password' not in str(rows)


def test_java_11_is_rejected(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('')
    monkeypatch.delenv('JAVA_HOME', raising=False)
    monkeypatch.setattr(preflight, 'run', lambda args, **kwargs: subprocess.CompletedProcess(args, 0, '', 'openjdk version "11.0.2"'))
    rows = {row['check']: row for row in preflight.check('spark', tmp_path)}
    assert not rows['Java']['ok']
    assert '>=17' in rows['Java']['message']


def test_empty_data_reports_bootstrap_requirement(tmp_path):
    (tmp_path / 'requirements.txt').write_text('')
    rows = preflight.check('unit', tmp_path, data=True)
    failures = [row for row in rows if row['check'].startswith('data/')]
    assert len(failures) == 3
    assert all(not row['ok'] and 'bootstrap' in row['message'] for row in failures)


def test_tracked_guard_includes_repository_bootstrap_configuration(tmp_path, monkeypatch):
    (tmp_path / 'README.md').write_text('')
    (tmp_path / 'docker-compose.yml').write_text('')
    (tmp_path / 'requirements.txt').write_text('')
    monkeypatch.setattr(preflight, 'run', lambda args, **kwargs: subprocess.CompletedProcess(args, 0, '', ''))
    rows = {row['check']: row for row in preflight.check('unit', tmp_path, tracked=True)}
    missing = rows['tracked files']['message']
    for required in ('.env.example', '.github/workflows/ci.yml', 'pytest.ini', 'sql/001_analytics.sql', 'airflow/matr_pipeline.py', 'dashboard/requirements.txt'):
        assert required in missing


@pytest.mark.parametrize('reported', ['2.0.0', 'v2.39.0', 'Docker Compose version v5.3.1'])
def test_current_and_newer_compose_versions_are_supported(reported):
    assert preflight.compose_version_supported(reported)


@pytest.mark.parametrize('reported', ['1.29.2', 'Docker Compose version v1.29.2', 'garbage'])
def test_legacy_or_unparseable_compose_versions_are_rejected(reported):
    assert not preflight.compose_version_supported(reported)


def test_host_checks_do_not_apply_container_lockfiles(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('psycopg[binary]>=3.2,<4\n')
    (tmp_path / 'requirements.survival-serving.txt').write_text('psycopg[binary]==3.2.9\nscikit-survival==0.24.1\n')
    monkeypatch.setattr(preflight.importlib.metadata, 'version', lambda name: '3.3.5')
    monkeypatch.setattr(preflight, 'run', lambda args, **kwargs: subprocess.CompletedProcess(args, 0, '', ''))
    rows = preflight.check('survival', tmp_path)
    psycopg = [row for row in rows if row['check'] == 'psycopg']
    assert psycopg == [{'check': 'psycopg', 'ok': True, 'message': '3.3.5; expected >=3.2,<4 from requirements.txt; install with python -m pip install -r requirements.txt'}]


def test_survival_container_pins_are_preserved():
    root = Path(__file__).resolve().parents[1]
    assert 'scikit-survival==0.24.1' in (root / 'requirements.survival-training.txt').read_text()
    assert 'scikit-survival==0.24.1' in (root / 'requirements.survival-serving.txt').read_text()
    assert 'psycopg[binary]==3.2.9' in (root / 'requirements.survival-serving.txt').read_text()
