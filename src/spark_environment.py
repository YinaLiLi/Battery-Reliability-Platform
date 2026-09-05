"""Keep local Spark workers on the driver's interpreter."""
import os
import sys


def configure_local_python(master):
    if master.startswith('local'):
        os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
