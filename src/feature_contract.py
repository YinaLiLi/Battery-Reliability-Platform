"""One causal feature contract shared by historical and streaming RUL paths."""

from collections import defaultdict
import math


RUL_FEATURES = (
    "cycle_index", "internal_resistance_in_ohm", "temperature_min_in_C", "temperature_max_in_C", "charge_time_in_s",
    "prior_discharge_capacity_in_Ah", "capacity_slope_10", "rolling_capacity_mean_10", "temperature_span_in_C",
    "charge_time_delta", "voltage_min_in_V", "voltage_max_in_V", "voltage_mean_in_V", "current_mean_in_A",
    "current_abs_max_in_A", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "capacity_fade_from_prior",
    "coulombic_efficiency", "early_cycle_capacity_delta",
)

# Mergeable sufficient statistics shared with Structured Streaming.
SUM_COUNT_FIELDS = {
    "voltage_in_V": ("_voltage_sum", "_voltage_count"),
    "current_in_A": ("_current_sum", "_current_count"),
}
MAX_FIELDS = {
    "source_time_in_s": "charge_time_in_s",
    "charge_capacity_in_Ah": "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah": "discharge_capacity_in_Ah",
    "internal_resistance_in_ohm": "internal_resistance_in_ohm",
}
MIN_FIELDS = {"temperature_in_C": "temperature_min_in_C", "voltage_in_V": "voltage_min_in_V"}
MAX_SAMPLE_FIELDS = {"temperature_in_C": "temperature_max_in_C", "voltage_in_V": "voltage_max_in_V"}
ABS_MAX_FIELDS = {"current_in_A": "current_abs_max_in_A"}


def finite_number(value):
    """Return a finite float or the contract's null representation."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _values(rows, field):
    return [value for row in rows if (value := finite_number(row.get(field))) is not None]


def aggregate_cycle_samples(rows):
    """Reduce one finalized cycle's telemetry with canonical null semantics."""
    rows = list(rows)
    if not rows:
        raise ValueError("cycle telemetry is required")
    first = rows[0]
    result = {"dataset": first.get("dataset"), "battery_id": first.get("battery_id"), "cycle_index": int(first["cycle_index"])}
    sequences = [row.get("replay_sequence") for row in rows if row.get("replay_sequence") is not None]
    if sequences:
        result["replay_sequence"] = int(max(sequences))
    for source, (sum_name, count_name) in SUM_COUNT_FIELDS.items():
        values = _values(rows, source)
        result[sum_name], result[count_name] = sum(values), len(values)
    for source, target in MAX_FIELDS.items():
        values = _values(rows, source)
        result[target] = max(values) if values else None
    for source, target in MIN_FIELDS.items():
        values = _values(rows, source)
        result[target] = min(values) if values else None
    for source, target in MAX_SAMPLE_FIELDS.items():
        values = _values(rows, source)
        result[target] = max(values) if values else None
    for source, target in ABS_MAX_FIELDS.items():
        values = _values(rows, source)
        result[target] = max(map(abs, values)) if values else None
    return render_cycle_aggregate(result)


def render_cycle_aggregate(row):
    """Render mergeable aggregate statistics as model-ready cycle inputs."""
    result = dict(row)
    for source, (sum_name, count_name) in SUM_COUNT_FIELDS.items():
        target = f"{source.rsplit('_in_', 1)[0]}_mean_in_{source.rsplit('_in_', 1)[1]}"
        total, count = finite_number(result.get(sum_name)), result.get(count_name)
        result[target] = total / int(count) if total is not None and count else None
    result["maximum_temperature_in_C"] = result.get("temperature_max_in_C")
    return result


def _slope(rows):
    points = [(finite_number(row.get("cycle_index")), finite_number(row.get("discharge_capacity_in_Ah"))) for row in rows]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    return None if denominator == 0 else sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def feature_row(current, history):
    """Compute model features from this and only prior finalized cycles."""
    prior = sorted(history, key=lambda row: int(row["cycle_index"]))
    previous = prior[-1] if prior else {}
    window = prior[-9:]
    capacities = [finite_number(row.get("discharge_capacity_in_Ah")) for row in window]
    capacities = [value for value in capacities if value is not None]
    discharge, charge = finite_number(current.get("discharge_capacity_in_Ah")), finite_number(current.get("charge_capacity_in_Ah"))
    prior_discharge = finite_number(previous.get("discharge_capacity_in_Ah"))
    first_discharge = finite_number(prior[0].get("discharge_capacity_in_Ah")) if prior else None
    minimum, maximum = finite_number(current.get("temperature_min_in_C")), finite_number(current.get("temperature_max_in_C"))
    charge_time, prior_charge_time = finite_number(current.get("charge_time_in_s")), finite_number(previous.get("charge_time_in_s"))
    return {
        **current,
        "prior_discharge_capacity_in_Ah": prior_discharge,
        "capacity_fade_from_prior": None if discharge is None or prior_discharge is None else discharge - prior_discharge,
        "rolling_capacity_mean_10": sum(capacities) / len(capacities) if capacities else None,
        "capacity_slope_10": _slope(window),
        "temperature_span_in_C": None if minimum is None or maximum is None else maximum - minimum,
        "charge_time_delta": None if charge_time is None or prior_charge_time is None else charge_time - prior_charge_time,
        "coulombic_efficiency": None if discharge is None or charge in (None, 0) else discharge / charge,
        "early_cycle_capacity_delta": None if discharge is None or first_discharge is None else discharge - first_discharge,
    }


def feature_rows(rows):
    """Build ordered causal rows independently for each battery."""
    by_battery = defaultdict(list)
    for row in rows:
        by_battery[row["battery_id"]].append(dict(row))
    output = []
    for battery_rows in by_battery.values():
        history = []
        for row in sorted(battery_rows, key=lambda item: int(item["cycle_index"])):
            output.append(feature_row(row, history))
            history.append(row)
    return output


def spark_cycle_aggregate_expressions(F, columns):
    """Return the contract's mergeable per-cycle aggregate expressions."""
    def aggregate(source, operation):
        return operation(source) if source in columns else F.lit(None).cast("double")

    expressions = [F.count("*").alias("event_count")]
    for source, (sum_name, count_name) in SUM_COUNT_FIELDS.items():
        expressions.extend((aggregate(source, F.sum).alias(sum_name), aggregate(source, F.count).alias(count_name)))
    for source, target in MAX_FIELDS.items():
        expressions.append(aggregate(source, F.max).alias(target))
    for source, target in MIN_FIELDS.items():
        expressions.append(aggregate(source, F.min).alias(target))
    for source, target in MAX_SAMPLE_FIELDS.items():
        expressions.append(aggregate(source, F.max).alias(target))
    for source, target in ABS_MAX_FIELDS.items():
        expressions.append((F.max(F.abs(source)) if source in columns else F.lit(None).cast("double")).alias(target))
    return expressions


def render_spark_cycle_aggregate(frame, F):
    """Render aggregate sufficient statistics using the same names as Python."""
    for source, (sum_name, count_name) in SUM_COUNT_FIELDS.items():
        target = f"{source.rsplit('_in_', 1)[0]}_mean_in_{source.rsplit('_in_', 1)[1]}"
        frame = frame.withColumn(target, F.when(F.col(count_name) > 0, F.col(sum_name) / F.col(count_name)))
    return frame.withColumn("maximum_temperature_in_C", F.col("temperature_max_in_C")).withColumn(
        "average_voltage_in_V", F.col("voltage_mean_in_V")
    )


def spark_causal_features(frame, F, Window):
    """Build the same causal features as ``feature_row`` with Spark windows."""
    window = Window.partitionBy("battery_id").orderBy("cycle_index")
    prior_nine = window.rowsBetween(-9, -1)
    prior_cycle = F.lag("cycle_index").over(window)
    return (frame
        .withColumn("prior_discharge_capacity_in_Ah", F.lag("discharge_capacity_in_Ah").over(window))
        .withColumn("capacity_fade_from_prior", F.col("discharge_capacity_in_Ah") - F.col("prior_discharge_capacity_in_Ah"))
        .withColumn("capacity_slope_10", F.regr_slope("discharge_capacity_in_Ah", "cycle_index").over(prior_nine))
        .withColumn("rolling_capacity_mean_10", F.avg("discharge_capacity_in_Ah").over(prior_nine))
        .withColumn("coulombic_efficiency", F.col("discharge_capacity_in_Ah") / F.col("charge_capacity_in_Ah"))
        .withColumn("temperature_span_in_C", F.col("temperature_max_in_C") - F.col("temperature_min_in_C"))
        .withColumn("charge_time_delta", F.col("charge_time_in_s") - F.lag("charge_time_in_s").over(window))
        .withColumn("early_cycle_capacity_delta", F.when(
            prior_cycle.isNotNull(),
            F.col("discharge_capacity_in_Ah") - F.first("discharge_capacity_in_Ah").over(window.rowsBetween(Window.unboundedPreceding, 0)),
        )))
