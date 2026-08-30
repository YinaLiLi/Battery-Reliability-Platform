"""Generate MVP EV telemetry from normalized NASA battery-aging references."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RAW_TELEMETRY_FIELDS = (
    "event_id",
    "vehicle_id",
    "timestamp",
    "battery_age_days",
    "battery_type",
    "region",
    "soc",
    "pack_voltage",
    "pack_current",
    "module_temp_min",
    "module_temp_max",
    "outside_temp",
    "odometer",
    "is_charging",
)
LABEL_FIELDS = ("event_id", "vehicle_id", "timestamp", "failure_within_30_operating_days")
DEFAULT_NASA_PATH = Path("data/processed/nasa_battery_cycles.parquet")


@dataclass(frozen=True)
class ReferenceProfile:
    positions: tuple[float, ...]
    retentions: tuple[float, ...]

    def retention_at(self, position):
        positions = np.asarray(self.positions)
        retentions = np.asarray(self.retentions)
        if position <= positions[-1]:
            return float(np.interp(position, positions, retentions))
        slope = (retentions[-1] - retentions[-2]) / (positions[-1] - positions[-2])
        return float(max(0.3, retentions[-1] + slope * (position - positions[-1])))


def build_reference_profiles(nasa_path=DEFAULT_NASA_PATH):
    """Return one normalized capacity-retention profile per NASA battery."""
    values = pq.read_table(nasa_path, columns=["battery_id", "discharge_cycle_number", "capacity_retention"]).to_pylist()
    grouped = {}
    for row in values:
        if row["capacity_retention"] is not None:
            grouped.setdefault(row["battery_id"], []).append(row)

    profiles = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row["discharge_cycle_number"])
        first, last = rows[0]["discharge_cycle_number"], rows[-1]["discharge_cycle_number"]
        span = max(last - first, 1.0)
        positions = tuple((row["discharge_cycle_number"] - first) / span for row in rows)
        retentions = tuple(float(row["capacity_retention"]) for row in rows)
        profiles.append(ReferenceProfile(positions, retentions))
    if len(profiles) < 2:
        raise ValueError("NASA input needs at least two capacity-retention profiles")
    return tuple(profiles)


def _retention(profiles, first_index, second_index, blend, position):
    first = profiles[first_index].retention_at(position)
    second = profiles[second_index].retention_at(position)
    return blend * first + (1 - blend) * second


def _outside_temperature(region, day, hour):
    base = {"south": 27.0, "west": 21.0, "midwest": 12.0}[region]
    return base + 7 * np.sin(2 * np.pi * day / 120) + 4 * np.sin(2 * np.pi * (hour - 8) / 24)


def simulate_fleet(
    nasa_path=DEFAULT_NASA_PATH,
    *,
    seed=42,
    vehicle_count=100,
    simulation_days=120,
    cadence_minutes=60,
    start_time=datetime(2025, 1, 1),
):
    """Return telemetry rows, offline labels, and validation metrics for a fleet."""
    if cadence_minutes != 60:
        raise ValueError("MVP simulator supports the planned 60-minute cadence only")

    profiles = build_reference_profiles(nasa_path)
    rng = np.random.default_rng(seed)
    telemetry, labels, final_retentions = [], [], []
    eol_vehicle_count = 0

    for vehicle_number in range(vehicle_count):
        vehicle_id = f"EV-{vehicle_number:04d}"
        region = ("south", "west", "midwest")[vehicle_number % 3]
        battery_type = ("standard", "long_range")[vehicle_number % 2]
        first_index, second_index = rng.choice(len(profiles), size=2, replace=False)
        blend = float(rng.uniform(0.7, 1.0))
        initial_position = float(rng.uniform(0.0, 0.65))
        progression_rate = float(rng.uniform(0.85, 1.15))
        daily_efc = float(rng.uniform(0.35, 0.75))
        battery_age_days = int(rng.integers(90, 1460))
        soc, odometer, efc = float(rng.uniform(55, 90)), float(rng.uniform(5_000, 95_000)), 0.0
        vehicle_events, failure_day = [], None

        for day in range(simulation_days):
            for hour in range(24):
                timestamp = start_time + timedelta(days=day, hours=hour)
                outside_temp = _outside_temperature(region, day, hour)
                is_driving = hour in (7, 8, 17, 18)
                is_charging = hour in (1, 2, 3) and soc < 88
                previous_soc = soc
                if is_driving:
                    soc = max(5.0, soc - daily_efc * 25)
                    pack_current = -float(rng.uniform(45, 110))
                    odometer += float(rng.uniform(8, 22))
                elif is_charging:
                    soc = min(92.0, soc + daily_efc * 100 / 3)
                    pack_current = float(rng.uniform(35, 85))
                else:
                    pack_current = float(rng.normal(0, 1.5))

                efc += abs(soc - previous_soc) / 200
                position = initial_position + progression_rate * efc / 120
                retention = _retention(profiles, first_index, second_index, blend, position)
                if retention <= 0.70:
                    failure_day = day + 1
                    break

                resistance_multiplier = 1 + 2 * (1 - retention)
                load_heat = abs(pack_current) * 0.025
                module_temp_min = outside_temp + load_heat * 0.55 + rng.normal(0, 0.5)
                module_temp_max = outside_temp + load_heat * (0.8 + 0.1 * resistance_multiplier) + rng.normal(0, 0.5)
                voltage = 320 + soc * 0.9 + min(pack_current, 0) * 0.14 * resistance_multiplier + rng.normal(0, 1.0)
                event = {
                    "event_id": f"{vehicle_id}-{timestamp:%Y%m%d%H%M}",
                    "vehicle_id": vehicle_id,
                    "timestamp": timestamp.isoformat(),
                    "battery_age_days": battery_age_days + day,
                    "battery_type": battery_type,
                    "region": region,
                    "soc": round(soc, 3),
                    "pack_voltage": round(voltage, 3),
                    "pack_current": round(pack_current, 3),
                    "module_temp_min": round(module_temp_min, 3),
                    "module_temp_max": round(max(module_temp_min, module_temp_max), 3),
                    "outside_temp": round(outside_temp, 3),
                    "odometer": round(odometer, 3),
                    "is_charging": is_charging,
                }
                telemetry.append(event)
                vehicle_events.append((event, day))
            if failure_day is not None:
                break

        if failure_day is not None:
            eol_vehicle_count += 1
        final_retentions.append(retention)
        for event, day in vehicle_events:
            if failure_day is not None and failure_day - day <= 30:
                target = 1
            elif day + 30 <= simulation_days:
                target = 0
            else:
                continue
            labels.append(
                {
                    "event_id": event["event_id"],
                    "vehicle_id": vehicle_id,
                    "timestamp": event["timestamp"],
                    "failure_within_30_operating_days": target,
                }
            )

    retention_array = np.asarray(final_retentions)
    positive_count = sum(row["failure_within_30_operating_days"] for row in labels)
    summary = {
        "vehicle_count": vehicle_count,
        "eol_vehicle_count": eol_vehicle_count,
        "fleet_eol_rate": eol_vehicle_count / vehicle_count if vehicle_count else 0.0,
        "positive_label_count": positive_count,
        "negative_label_count": len(labels) - positive_count,
        "retention_distribution": {f"p{percentile:02d}": float(np.percentile(retention_array, percentile)) for percentile in (5, 25, 50, 75, 95)},
    }
    return telemetry, labels, summary


def write_simulation(output_dir, telemetry, labels, summary):
    """Write simulator products without adding hidden health fields to telemetry."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": output_dir / "synthetic_fleet_telemetry.parquet",
        "labels": output_dir / "synthetic_fleet_labels.parquet",
        "validation": output_dir / "synthetic_fleet_validation.json",
    }
    pq.write_table(pa.Table.from_pylist(telemetry), paths["telemetry"])
    pq.write_table(pa.Table.from_pylist(labels, schema=pa.schema([(field, pa.string() if field != "failure_within_30_operating_days" else pa.int64()) for field in LABEL_FIELDS])), paths["labels"])
    paths["validation"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return paths


if __name__ == "__main__":
    write_simulation("data/processed", *simulate_fleet())
