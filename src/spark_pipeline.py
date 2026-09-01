"""Build leakage-safe cycle-level MATR degradation features with PySpark."""
import argparse
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

SHUFFLE_PARTITIONS = 3

def build_spark_session(master="spark://spark-master:7077"):
    return SparkSession.builder.master(master).appName("matr-degradation-features").config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS).getOrCreate()

def build_features(cycles, provenance, measurements=None):
    """Features use only current/prior cycles; SOH is measured and RUL is the ML target."""
    w = Window.partitionBy("battery_id").orderBy("cycle_index")
    frame=cycles.join(provenance.select("battery_id", "lineage_group_id", "batch_id", "charge_policy"), "battery_id")
    if measurements is not None:
        curves=measurements.groupBy('battery_id','cycle_index').agg(F.min('voltage_in_V').alias('voltage_min_in_V'),F.max('voltage_in_V').alias('voltage_max_in_V'),F.avg('voltage_in_V').alias('voltage_mean_in_V'),F.avg('current_in_A').alias('current_mean_in_A'),F.max(F.abs('current_in_A')).alias('current_abs_max_in_A'))
        frame=frame.join(curves,['battery_id','cycle_index'],'left')
    return (frame
        .withColumn("prior_discharge_capacity_in_Ah", F.lag("discharge_capacity_in_Ah").over(w))
        .withColumn("capacity_fade_from_prior", F.col("discharge_capacity_in_Ah") - F.col("prior_discharge_capacity_in_Ah"))
        .withColumn("capacity_slope_10", F.regr_slope("discharge_capacity_in_Ah", "cycle_index").over(w.rowsBetween(-9, -1)))
        .withColumn("rolling_capacity_mean_10", F.avg("discharge_capacity_in_Ah").over(w.rowsBetween(-9, -1)))
        .withColumn("coulombic_efficiency", F.col("discharge_capacity_in_Ah") / F.col("charge_capacity_in_Ah"))
        .withColumn("temperature_span_in_C", F.col("temperature_max_in_C") - F.col("temperature_min_in_C"))
        .withColumn("charge_time_delta", F.col("charge_time_in_s") - F.lag("charge_time_in_s").over(w))
        .withColumn("early_cycle_capacity_delta", F.col("discharge_capacity_in_Ah")-F.first("discharge_capacity_in_Ah").over(w.rowsBetween(Window.unboundedPreceding,0))))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--cycles",type=Path,default=Path("data/processed/matr/cycle_summary")); p.add_argument("--measurements",type=Path,default=Path("data/processed/matr/cycle_measurements")); p.add_argument("--provenance",type=Path,default=Path("data/processed/matr/matr_provenance.parquet")); p.add_argument("--output",type=Path,default=Path("data/processed/matr/degradation_features")); p.add_argument("--master",default="spark://spark-master:7077"); a=p.parse_args()
    spark=build_spark_session(a.master)
    try: build_features(spark.read.parquet(str(a.cycles)),spark.read.parquet(str(a.provenance)),spark.read.parquet(str(a.measurements))).write.mode("overwrite").parquet(str(a.output))
    finally: spark.stop()
if __name__ == '__main__': main()
