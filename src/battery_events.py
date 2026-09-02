"""Versioned Kafka contract for deterministic MATR historical replay."""

SCHEMA_VERSION = "1.0"
EVENT_FIELDS = ("event_id","dataset","battery_id","cycle_index","sample_index","source_time_in_s","replay_event_time","replay_sequence","voltage_in_V","current_in_A","temperature_in_C","charge_capacity_in_Ah","discharge_capacity_in_Ah","internal_resistance_in_ohm")

def event_from_measurement(row):
    event={field:row.get(field) for field in EVENT_FIELDS}
    if not all(event[key] is not None for key in ("event_id","dataset","battery_id","cycle_index","sample_index","replay_event_time")):
        raise ValueError("measurement does not satisfy the battery event contract")
    event["schema_version"]=SCHEMA_VERSION
    return event

def ordered_measurements(rows):
    return sorted(rows,key=lambda row:(row["battery_id"],row["cycle_index"],row["sample_index"]))
