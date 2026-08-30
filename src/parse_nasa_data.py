from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat


BATTERY_IDS = ("B0005", "B0006", "B0007", "B0018")
RAW_DIR = Path("data/raw/nasa_batteries")
OUTPUT_PATH = Path("data/processed/nasa_battery_cycles.parquet")
COLUMNS = (
    "battery_id",
    "discharge_cycle_number",
    "ambient_temperature",
    "capacity",
    "capacity_retention",
    "avg_voltage",
    "min_voltage",
    "max_voltage",
    "avg_current",
    "avg_temperature",
    "max_temperature",
    "Re",
    "Rct",
)


def field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def scalar(value):
    if value is None:
        return None
    array = np.asarray(value).squeeze()
    if array.size != 1:
        return None
    number = float(array.item())
    return number if np.isfinite(number) else None


def summary(value, function):
    if value is None:
        return None
    array = np.asarray(value, dtype=float).ravel()
    if not array.size or not np.isfinite(array).all():
        return None
    return float(function(array))


def parse_cycles(battery_id, cycles):
    rows = []
    last_re = last_rct = None
    first_capacity = None

    for cycle in np.atleast_1d(cycles):
        operation = field(cycle, "type")
        data = field(cycle, "data")

        if operation == "impedance":
            last_re, last_rct = scalar(field(data, "Re")), scalar(field(data, "Rct"))
        elif operation == "discharge":
            capacity = scalar(field(data, "Capacity"))
            if first_capacity is None and capacity is not None:
                first_capacity = capacity
            retention = None if capacity is None or first_capacity in (None, 0) else capacity / first_capacity
            rows.append(
                {
                    "battery_id": battery_id,
                    "discharge_cycle_number": len(rows) + 1,
                    "ambient_temperature": scalar(field(cycle, "ambient_temperature")),
                    "capacity": capacity,
                    "capacity_retention": retention,
                    "avg_voltage": summary(field(data, "Voltage_measured"), np.mean),
                    "min_voltage": summary(field(data, "Voltage_measured"), np.min),
                    "max_voltage": summary(field(data, "Voltage_measured"), np.max),
                    "avg_current": summary(field(data, "Current_measured"), np.mean),
                    "avg_temperature": summary(field(data, "Temperature_measured"), np.mean),
                    "max_temperature": summary(field(data, "Temperature_measured"), np.max),
                    "Re": last_re,
                    "Rct": last_rct,
                }
            )
    return rows


def parse_all(raw_dir=RAW_DIR, output_path=OUTPUT_PATH):
    rows = []
    for battery_id in BATTERY_IDS:
        battery = loadmat(raw_dir / f"{battery_id}.mat", squeeze_me=True, struct_as_record=False)[battery_id]
        rows.extend(parse_cycles(battery_id, field(battery, "cycle")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([(column, pa.string() if column == "battery_id" else pa.float64()) for column in COLUMNS])), output_path)
    return rows


if __name__ == "__main__":
    parse_all()
