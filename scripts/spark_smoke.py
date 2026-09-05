"""Exercise Java, Python workers and Spark together in a fresh image."""
from pyspark.sql import SparkSession

spark = SparkSession.builder.master('local[1]').appName('compatibility-smoke').getOrCreate()
try:
    assert spark.version == '4.1.3'
    assert spark.sparkContext.parallelize([1, 2, 3]).map(lambda n: n * 2).sum() == 12
finally:
    spark.stop()
