import sys
from src.spark_environment import configure_local_python


def test_local_workers_use_driver_unless_explicitly_configured(monkeypatch):
    monkeypatch.delenv('PYSPARK_PYTHON', raising=False)
    configure_local_python('spark://master:7077')
    import os
    assert 'PYSPARK_PYTHON' not in os.environ
    configure_local_python('local[1]')
    assert os.environ['PYSPARK_PYTHON'] == sys.executable
    monkeypatch.setenv('PYSPARK_PYTHON', '/explicit/python')
    configure_local_python('local[1]')
    assert os.environ['PYSPARK_PYTHON'] == '/explicit/python'
