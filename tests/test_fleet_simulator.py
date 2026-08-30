from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from src.fleet_simulator import (
    RAW_TELEMETRY_FIELDS,
    build_reference_profiles,
    simulate_fleet,
    write_simulation,
)


def nasa_parquet(tmp_path):
    rows = []
    for battery_id, endpoint in (("A", 0.68), ("B", 0.74)):
        for cycle, retention in enumerate((1.0, 0.9, endpoint), start=1):
            rows.append(
                {"battery_id": battery_id, "discharge_cycle_number": float(cycle), "capacity_retention": retention}
            )
    path = tmp_path / "nasa.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_simulator_is_deterministic_and_keeps_health_out_of_telemetry(tmp_path):
    nasa_path = nasa_parquet(tmp_path)

    first = simulate_fleet(nasa_path, seed=7, vehicle_count=8, simulation_days=45)
    second = simulate_fleet(nasa_path, seed=7, vehicle_count=8, simulation_days=45)

    telemetry, labels, summary = first
    assert first == second
    assert telemetry
    assert set(telemetry[0]) == set(RAW_TELEMETRY_FIELDS)
    assert {"true_capacity_retention", "equivalent_full_cycles", "failure_time"}.isdisjoint(telemetry[0])
    assert all(0 <= row["soc"] <= 100 for row in telemetry)
    assert summary["positive_label_count"] + summary["negative_label_count"] == len(labels)


def test_efc_alone_controls_retention_for_a_fixed_reference_profile(tmp_path):
    profiles = build_reference_profiles(nasa_parquet(tmp_path))

    retention = profiles[0].retention_at(position=0.5)
    assert retention == profiles[0].retention_at(position=0.5)
    assert retention != profiles[0].retention_at(position=0.75)


def test_eol_stops_a_vehicle_and_creates_positive_labels(tmp_path):
    telemetry, _, summary = simulate_fleet(nasa_parquet(tmp_path), seed=7, vehicle_count=8, simulation_days=120)
    event_counts = {}
    for event in telemetry:
        event_counts[event["vehicle_id"]] = event_counts.get(event["vehicle_id"], 0) + 1

    assert summary["eol_vehicle_count"] == 1
    assert summary["positive_label_count"] > 0
    assert min(event_counts.values()) < 120 * 24


def test_labels_have_a_complete_30_operating_day_horizon_and_outputs_are_written(tmp_path):
    nasa_path = nasa_parquet(tmp_path)
    telemetry, labels, summary = simulate_fleet(nasa_path, seed=2, vehicle_count=10, simulation_days=50)
    output_dir = tmp_path / "outputs"

    paths = write_simulation(output_dir, telemetry, labels, summary)

    assert all(row["failure_within_30_operating_days"] in (0, 1) for row in labels)
    assert all(
        row["failure_within_30_operating_days"] == 1 or datetime.fromisoformat(row["timestamp"]).date().isoformat() <= "2025-01-21"
        for row in labels
    )
    assert set(paths) == {"telemetry", "labels", "validation"}
    assert all(path.exists() for path in paths.values())
    assert {"eol_vehicle_count", "fleet_eol_rate", "retention_distribution"} <= set(summary)
