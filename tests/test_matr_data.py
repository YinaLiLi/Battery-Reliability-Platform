import json
import pickle

import pyarrow.parquet as pq

from src.matr_data import build_provenance, normalize_archive, normalize_cell


def _cell(cell_id="MATR_A"):
    return {
        "cell_id": cell_id,
        "nominal_capacity_in_Ah": 1.0,
        "SOC_interval": [0, 1],
        "charge_protocol": "4C",
        "cycle_data": [
            {
                "cycle_number": 1,
                "voltage_in_V": [3.0, 3.4],
                "current_in_A": [4.0, -4.0],
                "temperature_in_C": [25.0, 27.0],
                "charge_capacity_in_Ah": [0.0, 1.0],
                "discharge_capacity_in_Ah": [0.0, 0.95],
                "time_in_s": [0.0, 10.0],
            },
            {
                "cycle_number": 2,
                "voltage_in_V": [3.0, 3.4],
                "current_in_A": [4.0, -4.0],
                "temperature_in_C": [26.0, 28.0],
                "charge_capacity_in_Ah": [0.0, 0.9],
                "discharge_capacity_in_Ah": [0.0, 0.8],
                "time_in_s": [0.0, 12.0],
            },
        ],
    }


def test_normalize_cell_emits_cycle_targets_and_measurements():
    battery, cycles, measurements = normalize_cell(_cell(), eol_cycle=2, source_file="MATR_A.pkl")

    assert battery["battery_id"] == "MATR_A"
    assert [(row["cycle_index"], row["soh"], row["rul_cycles"]) for row in cycles] == [(1, 0.95, 1), (2, 0.8, 0)]
    assert measurements[1]["event_id"] == "matr:MATR_A:1:1"
    assert measurements[1]["internal_resistance_in_ohm"] is None


def test_normalize_cell_accepts_scalar_optional_resistance():
    cell = _cell()
    cell["cycle_data"][0]["internal_resistance_in_ohm"] = 0.02
    _, cycles, measurements = normalize_cell(cell, eol_cycle=2, source_file="MATR_A.pkl")
    assert cycles[0]["internal_resistance_in_ohm"] == 0.02
    assert measurements[0]["internal_resistance_in_ohm"] == 0.02


def test_normalize_cell_serializes_protocol_lists():
    cell = _cell()
    cell["charge_protocol"] = ["4C", "4.2V"]
    battery, _, _ = normalize_cell(cell, eol_cycle=2, source_file="MATR_A.pkl")
    assert battery["charge_protocol"] == '["4C", "4.2V"]'


def test_provenance_groups_explicit_continuations_and_identical_measurements():
    first = _cell("b1c0")
    continued = _cell("b2c7")
    unrelated = _cell("b3c1")
    rows = build_provenance(
        [("b1c0.pkl", first), ("b2c7.pkl", continued), ("b3c1.pkl", unrelated)],
        continuations={"b2c7": "b1c0"},
    )

    groups = {row["battery_id"]: row["lineage_group_id"] for row in rows}
    assert groups["b1c0"] == groups["b2c7"]
    assert groups["b3c1"] != groups["b1c0"]
    assert {row["provenance_status"] for row in rows} == {"verified", "documented_singleton"}


def test_normalize_archive_writes_three_canonical_parquet_tables(tmp_path):
    raw = tmp_path / "MATR"
    raw.mkdir()
    with (raw / "MATR_A.pkl").open("wb") as stream:
        pickle.dump(_cell(), stream)
    labels = tmp_path / "MATR_labels.json"
    labels.write_text(json.dumps({"MATR_A.pkl": 2}))

    outputs = normalize_archive(raw, labels, tmp_path / "processed")

    assert set(outputs) == {"battery_dim", "cycle_summary", "cycle_measurements", "matr_provenance"}
    assert pq.read_table(outputs["cycle_summary"]).num_rows == 2
    assert pq.read_table(outputs["cycle_measurements"]).num_rows == 4
