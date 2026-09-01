"""Memory-bounded quality checks for normalized MATR Parquet."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def files(path):
    return sorted(Path(path).glob("*.parquet"))


def scan(root):
    root = Path(root)
    battery_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files(root / "battery_dim"))
    battery_ids, cycle_keys, cycles, duplicates = set(), set(), [], 0
    for path in files(root / "cycle_summary"):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            key = (row["battery_id"], row["cycle_index"])
            duplicates += key in cycle_keys; cycle_keys.add(key)
            cycles.append(row)
            battery_ids.add(row["battery_id"])
    grouped = {}
    for row in cycles:
        grouped.setdefault(row["battery_id"], []).append(row)
    ordered = all([r["cycle_index"] for r in rows] == sorted(r["cycle_index"] for r in rows) for rows in grouped.values())
    soh = [r["soh"] for r in cycles if r["soh"] is not None]
    rul = [r["rul_cycles"] for r in cycles if r["rul_cycles"] is not None]
    measurement_rows = 0
    ordering_errors = missing_temp = missing_ir = 0
    ranges = {name: [None, None] for name in ("voltage_in_V", "current_in_A", "temperature_in_C")}
    for path in files(root / "cycle_measurements"):
        pf = pq.ParquetFile(path)
        measurement_rows += pf.metadata.num_rows
        last = None
        for batch in pf.iter_batches(columns=["battery_id", "cycle_index", "sample_index", "time_in_s", "voltage_in_V", "current_in_A", "temperature_in_C", "internal_resistance_in_ohm"]):
            for row in batch.to_pylist():
                key = (row["battery_id"], row["cycle_index"], row["sample_index"])
                # Canonical replay order is the source-array position (sample_index).
                # MATR contains harmless sub-microsecond floating-point time reversals.
                if last and key <= last[:3]: ordering_errors += 1
                last = (*key, row["time_in_s"])
                missing_temp += row["temperature_in_C"] is None
                missing_ir += row["internal_resistance_in_ohm"] is None
                for name, bounds in ranges.items():
                    value = row[name]
                    if value is not None:
                        bounds[0] = value if bounds[0] is None else min(bounds[0], value)
                        bounds[1] = value if bounds[1] is None else max(bounds[1], value)
    return {
        "battery_dim_rows": battery_rows, "unique_battery_ids": len(battery_ids), "cycle_summary_rows": len(cycles),
        "cycles_per_battery": {str(k): v for k, v in sorted(Counter(map(len, grouped.values())).items())},
        "cycle_key_duplicates": duplicates, "monotonic_cycles": ordered, "measurement_rows": measurement_rows,
        "measurement_ordering_errors": ordering_errors, "soh_range": [min(soh), max(soh)],
        "invalid_soh": sum(x <= 0 or x > 1.5 for x in soh), "rul_range": [min(rul), max(rul)],
        "negative_rul": sum(x < 0 for x in rul), "labelled_cycles": len(rul), "measurement_ranges": ranges,
        "missing_temperature": missing_temp, "missing_resistance": missing_ir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="data/processed/matr"); parser.add_argument("--output")
    args = parser.parse_args(); report = scan(args.root); print(json.dumps(report, indent=2, sort_keys=True))
    if args.output: Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
