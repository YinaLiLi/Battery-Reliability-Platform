"""Normalize BatteryLife MATR pickles into battery, cycle, and measurement rows."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    from .continuous_arrival import build_arrival_manifest
    from .matr_stage2 import lineage_split
except ImportError:
    from continuous_arrival import build_arrival_manifest
    from matr_stage2 import lineage_split


DATASET = "MATR"
# The five public Batch-1 experiments continued as Batch-2 experiments.
OFFICIAL_CONTINUATIONS = {
    "MATR_b2c7": "MATR_b1c0", "MATR_b2c8": "MATR_b1c1", "MATR_b2c9": "MATR_b1c2",
    "MATR_b2c15": "MATR_b1c3", "MATR_b2c16": "MATR_b1c4",
}

BATTERY_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("battery_id", pa.string()), ("source_file", pa.string()),
    ("nominal_capacity_in_Ah", pa.float64()), ("soc_start", pa.float64()), ("soc_end", pa.float64()),
    ("charge_protocol", pa.string()), ("discharge_protocol", pa.string()), ("form_factor", pa.string()),
    ("anode_material", pa.string()), ("cathode_material", pa.string()),
])
SUMMARY_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("battery_id", pa.string()), ("cycle_index", pa.int32()),
    ("eol_cycle", pa.int32()), ("rul_cycles", pa.int32()), ("charge_capacity_in_Ah", pa.float64()),
    ("discharge_capacity_in_Ah", pa.float64()), ("soh", pa.float64()),
    ("internal_resistance_in_ohm", pa.float64()), ("temperature_min_in_C", pa.float64()),
    ("temperature_max_in_C", pa.float64()), ("charge_time_in_s", pa.float64()),
])
MEASUREMENT_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("dataset", pa.string()), ("battery_id", pa.string()),
    ("cycle_index", pa.int32()), ("sample_index", pa.int32()), ("source_time_in_s", pa.float64()),
    ("replay_event_time", pa.string()), ("time_in_s", pa.float64()), ("voltage_in_V", pa.float64()),
    ("current_in_A", pa.float64()), ("temperature_in_C", pa.float64()),
    ("charge_capacity_in_Ah", pa.float64()), ("discharge_capacity_in_Ah", pa.float64()),
    ("internal_resistance_in_ohm", pa.float64()),
])
PROVENANCE_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("battery_id", pa.string()), ("source_file", pa.string()),
    ("batch_id", pa.string()), ("lineage_group_id", pa.string()), ("original_battery_id", pa.string()),
    ("provenance_status", pa.string()), ("lineage_reason", pa.string()), ("fingerprint", pa.string()),
    ("charge_policy", pa.string()),
])


def _values(cycle, field):
    value = cycle.get(field)
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def _at(values, index):
    return values[index] if index < len(values) else None


def _maximum(values):
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _text(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else (str(value) if value is not None else None)


def normalize_cell(cell, *, eol_cycle, source_file):
    """Return normalized battery, cycle-summary, and measurement records for one cell."""
    battery_id = str(cell.get("cell_id") or Path(source_file).stem)
    soc = cell.get("SOC_interval") or [0, 1]
    soc_width = float(soc[1] - soc[0]) or 1.0
    nominal = float(cell.get("nominal_capacity_in_Ah"))
    battery = {
        "dataset": DATASET,
        "battery_id": battery_id,
        "source_file": source_file,
        "nominal_capacity_in_Ah": nominal,
        "soc_start": float(soc[0]),
        "soc_end": float(soc[1]),
        "charge_protocol": _text(cell.get("charge_protocol")),
        "discharge_protocol": _text(cell.get("discharge_protocol")),
        "form_factor": _text(cell.get("form_factor")),
        "anode_material": _text(cell.get("anode_material")),
        "cathode_material": _text(cell.get("cathode_material")),
    }
    summaries, measurements = [], []
    for fallback_index, cycle in enumerate(cell.get("cycle_data") or [], start=1):
        cycle_index = int(cycle.get("cycle_number") or fallback_index)
        discharge = _values(cycle, "discharge_capacity_in_Ah")
        charge = _values(cycle, "charge_capacity_in_Ah")
        capacity = _maximum(discharge)
        summaries.append(
            {
                "dataset": DATASET,
                "battery_id": battery_id,
                "cycle_index": cycle_index,
                "eol_cycle": eol_cycle,
                "rul_cycles": max(eol_cycle - cycle_index, 0) if eol_cycle is not None else None,
                "charge_capacity_in_Ah": _maximum(charge),
                "discharge_capacity_in_Ah": capacity,
                "soh": capacity / nominal / soc_width if capacity is not None else None,
                "internal_resistance_in_ohm": _maximum(_values(cycle, "internal_resistance_in_ohm")),
                "temperature_min_in_C": min(_values(cycle, "temperature_in_C"), default=None),
                "temperature_max_in_C": _maximum(_values(cycle, "temperature_in_C")),
                "charge_time_in_s": max(_values(cycle, "time_in_s"), default=None),
            }
        )
        columns = {name: _values(cycle, name) for name in (
            "time_in_s", "voltage_in_V", "current_in_A", "temperature_in_C", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "internal_resistance_in_ohm"
        )}
        for sample_index in range(max(map(len, columns.values()), default=0)):
            measurements.append(
                {
                    "event_id": f"matr:{battery_id}:{cycle_index}:{sample_index}",
                    "dataset": DATASET,
                    "battery_id": battery_id,
                    "cycle_index": cycle_index,
                    "sample_index": sample_index,
                    "source_time_in_s": _at(columns["time_in_s"], sample_index),
                    "replay_event_time": (
                        datetime(2020, 1, 1, tzinfo=timezone.utc)
                        + timedelta(days=cycle_index - 1, seconds=float(_at(columns["time_in_s"], sample_index) or 0))
                    ).isoformat(),
                    **{name: _at(values, sample_index) for name, values in columns.items()},
                }
            )
    return battery, summaries, measurements


def _fingerprint(raw_bytes):
    """Exact source-file fingerprint: duplicates share a scientific lineage."""
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def build_provenance(cells, *, continuations=None):
    """Build a conservative lineage manifest from known continuation identities."""
    continuations = OFFICIAL_CONTINUATIONS if continuations is None else continuations
    rows, fingerprints = [], {}
    for source_file, cell in cells:
        battery_id = str(cell.get("cell_id") or Path(source_file).stem)
        fingerprint = cell.get("_source_fingerprint") or _fingerprint(pickle.dumps(cell, protocol=4))
        root = continuations.get(battery_id, battery_id)
        if battery_id in continuations:
            lineage, reason, status = f"matr:{root}", "official_continuation", "verified"
        elif fingerprint in fingerprints:
            lineage, reason, status = fingerprints[fingerprint], "exact_source_duplicate", "verified"
        else:
            lineage, reason, status = f"matr:{root}", "documented_singleton", "documented_singleton"
        fingerprints.setdefault(fingerprint, lineage)
        token = next((part for part in Path(source_file).stem.split("_") if part.startswith("b") and "c" in part), None)
        batch = f"batch_{token[1]}" if token else None
        rows.append(
            {
                "dataset": DATASET,
                "battery_id": battery_id,
                "source_file": source_file,
                "batch_id": batch,
                "lineage_group_id": lineage,
                "original_battery_id": root,
                "provenance_status": status,
                "lineage_reason": reason,
                "fingerprint": fingerprint,
                "charge_policy": _text(cell.get("charge_protocol")),
            }
        )
    return rows


def _write_rows(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def normalize_archive(raw_dir, labels_path, output_dir, *, continuations=None):
    """Normalize a directory of official MATR pickle files and life labels."""
    raw_dir, labels_path, output_dir = Path(raw_dir), Path(labels_path), Path(output_dir)
    with labels_path.open() as stream:
        labels = json.load(stream)
    loaded = []
    outputs = {name: output_dir / name for name in ("battery_dim", "cycle_summary", "cycle_measurements")}
    for path in sorted(raw_dir.glob("*.pkl")):
        raw_bytes = path.read_bytes()
        cell = pickle.loads(raw_bytes)
        cell["_source_fingerprint"] = _fingerprint(raw_bytes)
        loaded.append((path.name, cell))
        battery, summary, samples = normalize_cell(cell, eol_cycle=labels.get(path.name), source_file=path.name)
        _write_rows(outputs["battery_dim"] / f"part-{len(loaded):04d}.parquet", [battery], BATTERY_SCHEMA)
        _write_rows(outputs["cycle_summary"] / f"part-{len(loaded):04d}.parquet", summary, SUMMARY_SCHEMA)
        _write_rows(outputs["cycle_measurements"] / f"part-{len(loaded):04d}.parquet", samples, MEASUREMENT_SCHEMA)
    if not loaded:
        raise ValueError(f"no MATR pickle files in {raw_dir}")
    provenance = build_provenance(loaded, continuations=continuations)
    outputs["matr_provenance"] = _write_rows(output_dir / "matr_provenance.parquet", provenance, PROVENANCE_SCHEMA)
    manifest = build_arrival_manifest(
        provenance,
        ds.dataset(outputs["cycle_summary"], format="parquet").to_table(columns=["battery_id", "cycle_index", "eol_cycle"]).to_pylist(),
        lineage_split(provenance),
    )
    outputs["arrival_manifest"] = output_dir / "arrival_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest), outputs["arrival_manifest"])
    return outputs
