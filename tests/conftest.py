import pytest
from src.spark_environment import configure_local_python


def pytest_sessionstart(session):
    if session.config.getoption('--tier') in ('spark', 'all'):
        configure_local_python('local[1]')


def pytest_addoption(parser):
    parser.addoption('--tier', choices=['unit', 'spark', 'survival', 'airflow', 'integration', 'all'], default='unit')


def tier(path):
    name = path.name
    if name == 'test_airflow_dag.py':
        return 'airflow'
    if name in ('test_spark_streaming.py', 'test_matr_spark_pipeline.py', 'test_postgres_loader.py'):
        return 'spark'
    if name == 'test_survival_runtime.py':
        return 'survival'
    if name == 'test_docker_integration.py':
        return 'integration'
    return 'unit'


def pytest_ignore_collect(collection_path, config):
    chosen = config.getoption('--tier')
    if collection_path.name.startswith('test_') and collection_path.suffix == '.py':
        return chosen != 'all' and tier(collection_path) != chosen


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(getattr(pytest.mark, tier(item.path)))


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    if reporter and reporter.stats.get('skipped'):
        session.exitstatus = 1
