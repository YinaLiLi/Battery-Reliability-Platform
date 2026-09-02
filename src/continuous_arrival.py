"""Deterministic arrival, label-validity, and retraining eligibility rules."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json


ARRIVAL_SCHEDULE_VERSION = "matr-arrival-v1"
LABEL_CLASSIFICATION_VERSION = "source-cycle-endpoint-v1"
SNAPSHOT_THRESHOLDS = (26, 51, 76, 94)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_arrival_manifest(provenance, cycle_rows, splits, *, start_epoch=None):
    """Build a replay-stable manifest without treating absent source cycles as EOL."""
    start_epoch = start_epoch or datetime(2020, 1, 1, tzinfo=timezone.utc)
    if start_epoch.tzinfo is None:
        raise ValueError("start_epoch must be timezone-aware")
    summaries = {}
    for row in cycle_rows:
        battery = row["battery_id"]
        current = summaries.setdefault(battery, {"last_source_cycle": 0, "eol_cycle": row.get("eol_cycle")})
        current["last_source_cycle"] = max(current["last_source_cycle"], int(row["cycle_index"]))
        if current["eol_cycle"] != row.get("eol_cycle"):
            raise ValueError(f"{battery} has inconsistent EOL labels")
    split_by_battery = {battery: split for split, batteries in splits.items() for battery in batteries}
    ordered = sorted(
        provenance,
        key=lambda row: (sha256(f"{ARRIVAL_SCHEDULE_VERSION}:{row['lineage_group_id']}".encode()).hexdigest(), row["battery_id"]),
    )
    provisional = []
    for rank, source in enumerate(ordered):
        battery = source["battery_id"]
        summary = summaries[battery]
        eol_cycle = summary["eol_cycle"]
        delta = None if eol_cycle is None else int(eol_cycle) - summary["last_source_cycle"]
        valid = eol_cycle is not None and delta <= 0
        provisional.append(
            {
                "dataset": source.get("dataset", "MATR"),
                "battery_id": battery,
                "lineage_group_id": source["lineage_group_id"],
                "split": split_by_battery[battery],
                "arrival_rank": rank,
                "start_time": (start_epoch + timedelta(days=rank)).isoformat(),
                "first_source_cycle": 1,
                "last_source_cycle": summary["last_source_cycle"],
                "eol_cycle": eol_cycle,
                "eol_cycle_delta": delta,
                "label_status": "valid_observed_endpoint" if valid else "unverified_endpoint_after_source_end",
                "valid_eol_label": valid,
                "arrival_schedule_version": ARRIVAL_SCHEDULE_VERSION,
                "label_classification_version": LABEL_CLASSIFICATION_VERSION,
            }
        )
    fingerprint = _digest(provisional)
    return [{**row, "schedule_fingerprint": fingerprint} for row in provisional]


def schedule_measurement(row, manifest_row, *, replay_sequence):
    """Assign simulated event time without changing the source experiment clock."""
    start = datetime.fromisoformat(manifest_row["start_time"])
    offset_days = int(row["cycle_index"]) - int(manifest_row["first_source_cycle"])
    source_seconds = float(row.get("source_time_in_s") or 0)
    return {
        **row,
        "replay_event_time": (start + timedelta(days=offset_days, seconds=source_seconds)).isoformat(),
        "replay_sequence": replay_sequence,
    }


def lifecycle_events_for_manifest(manifest_row):
    """Create independent EOL and source-replay completion facts for one battery."""
    start = datetime.fromisoformat(manifest_row["start_time"])

    def event(event_type, cycle):
        event_time = start + timedelta(days=int(cycle) - int(manifest_row["first_source_cycle"]))
        return {
            "event_id": f"matr-lifecycle:{manifest_row['battery_id']}:{event_type}",
            "event_type": event_type,
            "dataset": manifest_row.get("dataset", "MATR"),
            "battery_id": manifest_row["battery_id"],
            "cycle_index": int(cycle),
            "replay_event_time": event_time.isoformat(),
            "schema_version": "1.0",
        }

    events = []
    if manifest_row["valid_eol_label"]:
        events.append(event("eol_observed", manifest_row["eol_cycle"]))
    events.append(event("replay_complete", manifest_row["last_source_cycle"]))
    return events


def lifecycle_state(lifecycle_events):
    """Reduce immutable lifecycle events to the three explicit eligibility states."""
    state = {}
    for event in lifecycle_events:
        battery = event["battery_id"]
        current = state.setdefault(battery, {"eol_observed": False, "replay_complete": False})
        if "event_type" not in event:
            current["eol_observed"] = current["eol_observed"] or bool(event.get("eol_observed"))
            current["replay_complete"] = current["replay_complete"] or bool(event.get("replay_complete"))
        elif event["event_type"] == "eol_observed":
            current["eol_observed"] = True
        elif event["event_type"] == "replay_complete":
            current["replay_complete"] = True
    return state


def eligible_training_batteries(manifest, lifecycle_events):
    states = lifecycle_state(lifecycle_events)
    return {
        row["battery_id"]
        for row in manifest
        if row["split"] == "train"
        and row["valid_eol_label"]
        and states.get(row["battery_id"], {}).get("eol_observed")
        and states.get(row["battery_id"], {}).get("replay_complete")
    }


def next_snapshot(eligible_batteries, published_counts):
    """Return the next valid-battery threshold, or None when no new generation is due."""
    eligible = list(eligible_batteries)
    for threshold in SNAPSHOT_THRESHOLDS:
        if threshold not in published_counts and len(eligible) >= threshold:
            return threshold, eligible[:threshold]
    return None


def model_fingerprint(battery_ids, *, manifest_fingerprint, split_version, feature_version, model_config):
    return _digest(
        {
            "arrival_schedule_version": ARRIVAL_SCHEDULE_VERSION,
            "eligible_battery_ids": sorted(battery_ids),
            "manifest_fingerprint": manifest_fingerprint,
            "split_version": split_version,
            "feature_version": feature_version,
            "model_config": model_config,
        }
    )
