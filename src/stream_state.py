"""Content-addressed stream-state contracts shared by Streaming and Airflow."""

from collections.abc import Mapping
from hashlib import sha256
import json


STATE_SCHEMA_VERSION = "stream-state-v1"
BOUNDARY_SCHEMA_VERSION = "finalized-cycle-boundary-v1"
COMPACT_BOUNDARY_SCHEMA_VERSION = "finalized-cycle-boundary-v2"


class StreamStateValidationError(ValueError):
    """A state manifest or its authoritative finalized boundary is invalid."""


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise StreamStateValidationError(f"{name} is required")
    return value


def _cycle_key(row):
    if not isinstance(row, Mapping):
        raise StreamStateValidationError("finalized cycle key must be a mapping")
    dataset = _text(row.get("dataset"), "dataset")
    battery_id = _text(row.get("battery_id"), "battery_id")
    cycle_index = row.get("cycle_index")
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or cycle_index < 1:
        raise StreamStateValidationError("cycle_index must be a positive integer")
    return {"dataset": dataset, "battery_id": battery_id, "cycle_index": cycle_index}


def _key_tuple(row):
    return row["dataset"], row["battery_id"], row["cycle_index"]


def _normalized_keys(keys):
    if keys is None:
        raise StreamStateValidationError("finalized cycle keys are required")
    normalized = sorted((_cycle_key(row) for row in keys), key=lambda row: (row["dataset"], row["battery_id"], row["cycle_index"]))
    if len({_key_tuple(row) for row in normalized}) != len(normalized):
        raise StreamStateValidationError("finalized cycle keys must be unique")
    return normalized


def _normalized_ranges(ranges):
    if ranges is None:
        raise StreamStateValidationError("finalized cycle ranges are required")
    normalized = []
    for row in ranges:
        if not isinstance(row, Mapping):
            raise StreamStateValidationError("finalized cycle range must be a mapping")
        dataset = _text(row.get("dataset"), "dataset")
        battery_id = _text(row.get("battery_id"), "battery_id")
        maximum, count, sequence = (row.get(name) for name in (
            "max_finalized_cycle_index", "finalized_cycle_count", "replay_sequence"
        ))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (maximum, count)):
            raise StreamStateValidationError("compact boundary cycle indexes and counts must be positive integers")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise StreamStateValidationError("compact boundary replay_sequence must be a non-negative integer")
        normalized.append({"dataset": dataset, "battery_id": battery_id,
                           "max_finalized_cycle_index": maximum, "finalized_cycle_count": count,
                           "replay_sequence": sequence})
    normalized.sort(key=lambda row: (row["dataset"], row["battery_id"]))
    keys = [(row["dataset"], row["battery_id"]) for row in normalized]
    if len(keys) != len(set(keys)):
        raise StreamStateValidationError("finalized cycle ranges must be unique by dataset and battery")
    return normalized


def _normalized_battery_ids(battery_ids):
    if battery_ids is None:
        raise StreamStateValidationError("eligible completed training batteries are required")
    return sorted({_text(battery_id, "eligible completed training battery") for battery_id in battery_ids})


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise StreamStateValidationError(f"{name} is required")
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise StreamStateValidationError(f"{name} must be JSON serializable") from error
    return dict(value)


def _kafka_offsets(value):
    value = _mapping(value, "kafka offsets")
    normalized = {}
    for topic, partitions in value.items():
        topic = _text(topic, "Kafka topic")
        if not isinstance(partitions, Mapping):
            raise StreamStateValidationError("Kafka topic offsets must be a mapping")
        normalized[topic] = {}
        for partition, offset in partitions.items():
            try:
                partition = int(partition)
            except (TypeError, ValueError) as error:
                raise StreamStateValidationError("Kafka partition must be an integer") from error
            if partition < 0 or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise StreamStateValidationError("Kafka offsets must be non-negative integers")
            normalized[topic][str(partition)] = offset
    return {topic: {partition: normalized[topic][partition] for partition in sorted(normalized[topic], key=int)} for topic in sorted(normalized)}


def _boundary_payload(*, canonical_fingerprint, arrival_manifest_fingerprint, feature_contract_version, finalized_cycle_keys):
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "canonical_fingerprint": _text(canonical_fingerprint, "canonical fingerprint"),
        "arrival_manifest_fingerprint": _text(arrival_manifest_fingerprint, "arrival manifest fingerprint"),
        "feature_contract_version": _text(feature_contract_version, "feature contract version"),
        "finalized_cycle_keys": _normalized_keys(finalized_cycle_keys),
    }


def build_finalized_cycle_boundary(finalized_cycle_keys, *, canonical_fingerprint, arrival_manifest_fingerprint, feature_contract_version):
    """Build the canonical, content-addressed allowlist for one Stream state."""
    payload = _boundary_payload(
        canonical_fingerprint=canonical_fingerprint,
        arrival_manifest_fingerprint=arrival_manifest_fingerprint,
        feature_contract_version=feature_contract_version,
        finalized_cycle_keys=finalized_cycle_keys,
    )
    return {**payload, "boundary_fingerprint": _digest(payload)}


def build_compact_finalized_cycle_boundary(finalized_cycle_rows, *, canonical_cycle_keys,
                                           canonical_fingerprint, arrival_manifest_fingerprint,
                                           feature_contract_version):
    """Build one watermark per battery after proving its finalized keys form a canonical prefix."""
    canonical = _normalized_keys(canonical_cycle_keys)
    finalized = []
    for row in finalized_cycle_rows:
        key = _cycle_key(row)
        sequence = row.get("replay_sequence", 0)
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise StreamStateValidationError("finalized cycle replay_sequence must be a non-negative integer")
        finalized.append({**key, "replay_sequence": sequence})
    keys = [_key_tuple(row) for row in finalized]
    if len(keys) != len(set(keys)):
        raise StreamStateValidationError("finalized cycle keys must be unique")
    canonical_by_battery = {}
    for row in canonical:
        canonical_by_battery.setdefault((row["dataset"], row["battery_id"]), []).append(row["cycle_index"])
    finalized_by_battery = {}
    for row in finalized:
        finalized_by_battery.setdefault((row["dataset"], row["battery_id"]), []).append(row)
    ranges = []
    for battery, rows in finalized_by_battery.items():
        rows.sort(key=lambda row: row["cycle_index"])
        maximum = rows[-1]["cycle_index"]
        expected = [cycle for cycle in canonical_by_battery.get(battery, ()) if cycle <= maximum]
        actual = [row["cycle_index"] for row in rows]
        if actual != expected:
            raise StreamStateValidationError(
                f"finalized cycles must be prefix-complete for {battery[0]}/{battery[1]}"
            )
        ranges.append({"dataset": battery[0], "battery_id": battery[1],
                       "max_finalized_cycle_index": maximum, "finalized_cycle_count": len(actual),
                       "replay_sequence": rows[-1]["replay_sequence"]})
    payload = {
        "schema_version": COMPACT_BOUNDARY_SCHEMA_VERSION,
        "canonical_fingerprint": _text(canonical_fingerprint, "canonical fingerprint"),
        "arrival_manifest_fingerprint": _text(arrival_manifest_fingerprint, "arrival manifest fingerprint"),
        "feature_contract_version": _text(feature_contract_version, "feature contract version"),
        "finalized_cycle_ranges": _normalized_ranges(ranges),
    }
    return {**payload, "boundary_fingerprint": _digest(payload)}


def validate_finalized_cycle_boundary(boundary):
    """Return a validated boundary or reject a missing/tampered artifact."""
    if boundary is None:
        raise StreamStateValidationError("finalized cycle boundary is required")
    if not isinstance(boundary, Mapping):
        raise StreamStateValidationError("finalized cycle boundary must be a mapping")
    if boundary.get("schema_version") == COMPACT_BOUNDARY_SCHEMA_VERSION:
        payload = {
            "schema_version": COMPACT_BOUNDARY_SCHEMA_VERSION,
            "canonical_fingerprint": _text(boundary.get("canonical_fingerprint"), "canonical fingerprint"),
            "arrival_manifest_fingerprint": _text(boundary.get("arrival_manifest_fingerprint"), "arrival manifest fingerprint"),
            "feature_contract_version": _text(boundary.get("feature_contract_version"), "feature contract version"),
            "finalized_cycle_ranges": _normalized_ranges(boundary.get("finalized_cycle_ranges")),
        }
        if list(boundary.get("finalized_cycle_ranges", ())) != payload["finalized_cycle_ranges"]:
            raise StreamStateValidationError("finalized cycle ranges must use canonical order")
        if boundary.get("boundary_fingerprint") != _digest(payload):
            raise StreamStateValidationError("finalized cycle boundary fingerprint mismatch")
        return {**payload, "boundary_fingerprint": boundary["boundary_fingerprint"]}
    if boundary.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        raise StreamStateValidationError("unsupported finalized cycle boundary schema")
    payload = _boundary_payload(
        canonical_fingerprint=boundary.get("canonical_fingerprint"),
        arrival_manifest_fingerprint=boundary.get("arrival_manifest_fingerprint"),
        feature_contract_version=boundary.get("feature_contract_version"),
        finalized_cycle_keys=boundary.get("finalized_cycle_keys"),
    )
    if list(boundary.get("finalized_cycle_keys", ())) != payload["finalized_cycle_keys"]:
        raise StreamStateValidationError("finalized cycle keys must use canonical order")
    if boundary.get("boundary_fingerprint") != _digest(payload):
        raise StreamStateValidationError("finalized cycle boundary fingerprint mismatch")
    return {**payload, "boundary_fingerprint": boundary["boundary_fingerprint"]}


def _state_id(boundary):
    identity = {field: boundary[field] for field in (
        "canonical_fingerprint", "arrival_manifest_fingerprint", "feature_contract_version"
    )}
    identity["finalized_cycle_ranges" if "finalized_cycle_ranges" in boundary else "finalized_cycle_keys"] = (
        boundary.get("finalized_cycle_ranges", boundary.get("finalized_cycle_keys"))
    )
    return "stream-state-" + _digest(identity)


def finalized_cycle_count(boundary):
    boundary = validate_finalized_cycle_boundary(boundary)
    return (sum(row["finalized_cycle_count"] for row in boundary["finalized_cycle_ranges"])
            if "finalized_cycle_ranges" in boundary else len(boundary["finalized_cycle_keys"]))


def state_id_for_boundary(boundary):
    """Return the stable identity for a validated finalized-cycle boundary."""
    return _state_id(validate_finalized_cycle_boundary(boundary))


def create_stream_state_manifest(boundary, *, boundary_ref, eligible_completed_training_batteries, cutoff_metadata, kafka_offsets):
    """Create an immutable-by-content manifest after all state artifacts are complete."""
    boundary = validate_finalized_cycle_boundary(boundary)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "state_id": _state_id(boundary),
        "canonical_fingerprint": boundary["canonical_fingerprint"],
        "arrival_manifest_fingerprint": boundary["arrival_manifest_fingerprint"],
        "feature_contract_version": boundary["feature_contract_version"],
        "finalized_cycle_boundary_ref": _text(boundary_ref, "finalized cycle boundary ref"),
        "finalized_cycle_boundary_fingerprint": boundary["boundary_fingerprint"],
        "finalized_cycle_count": finalized_cycle_count(boundary),
        "eligible_completed_training_batteries": _normalized_battery_ids(eligible_completed_training_batteries),
        "cutoff_metadata": _mapping(cutoff_metadata, "cutoff metadata"),
        "kafka_offsets": _kafka_offsets(kafka_offsets),
    }


def validate_stream_state_manifest(manifest, boundary, *, expected_canonical_fingerprint, expected_arrival_manifest_fingerprint=None, expected_feature_contract_version=None):
    """Validate the manifest and boundary Airflow must use as its exact as-of input."""
    if not isinstance(manifest, Mapping):
        raise StreamStateValidationError("stream state manifest is required")
    if manifest.get("schema_version") != STATE_SCHEMA_VERSION:
        raise StreamStateValidationError("unsupported stream state manifest schema")
    boundary = validate_finalized_cycle_boundary(boundary)
    expected = {
        "canonical_fingerprint": _text(expected_canonical_fingerprint, "canonical fingerprint"),
        "arrival_manifest_fingerprint": expected_arrival_manifest_fingerprint,
        "feature_contract_version": expected_feature_contract_version,
    }
    for field, expected_value in expected.items():
        if expected_value is not None and manifest.get(field) != expected_value:
            raise StreamStateValidationError(f"{field.replace('_', ' ')} mismatch")
        if manifest.get(field) != boundary.get(field):
            raise StreamStateValidationError(f"{field.replace('_', ' ')} mismatch between manifest and boundary")
    if manifest.get("finalized_cycle_boundary_fingerprint") != boundary["boundary_fingerprint"]:
        raise StreamStateValidationError("finalized cycle boundary fingerprint mismatch")
    if manifest.get("finalized_cycle_count") != finalized_cycle_count(boundary):
        raise StreamStateValidationError("finalized cycle count mismatch")
    if manifest.get("state_id") != _state_id(boundary):
        raise StreamStateValidationError("stream state id mismatch")
    if manifest.get("eligible_completed_training_batteries") != _normalized_battery_ids(manifest.get("eligible_completed_training_batteries")):
        raise StreamStateValidationError("eligible completed training batteries must use canonical order")
    _text(manifest.get("finalized_cycle_boundary_ref"), "finalized cycle boundary ref")
    _mapping(manifest.get("cutoff_metadata"), "cutoff metadata")
    if manifest.get("kafka_offsets") != _kafka_offsets(manifest.get("kafka_offsets")):
        raise StreamStateValidationError("Kafka offsets must use canonical order")
    shared_fields = ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")
    if any(field in manifest for field in shared_fields):
        if not all(field in manifest for field in shared_fields):
            raise StreamStateValidationError("shared generation cohort is incomplete")
        arrived, observed, censored = (list(manifest[field]) for field in shared_fields)
        if any(not isinstance(value, str) or not value for values in (arrived, observed, censored) for value in values):
            raise StreamStateValidationError("shared generation cohorts require battery ids")
        if len(arrived) != len(set(arrived)) or len(observed) != len(set(observed)) or len(censored) != len(set(censored)):
            raise StreamStateValidationError("shared generation cohorts must be unique")
        if set(observed) | set(censored) != set(arrived) or set(observed) & set(censored):
            raise StreamStateValidationError("shared generation cohorts must partition arrived training batteries")
    return dict(manifest)


def select_finalized_cycle_rows(rows, boundary):
    """Select rows solely by the authoritative finalized-cycle allowlist."""
    boundary = validate_finalized_cycle_boundary(boundary)
    if "finalized_cycle_ranges" in boundary:
        maximum = {(row["dataset"], row["battery_id"]): row["max_finalized_cycle_index"]
                   for row in boundary["finalized_cycle_ranges"]}
        return [row for row in rows if _cycle_key(row)["cycle_index"] <= maximum.get(
            (_cycle_key(row)["dataset"], _cycle_key(row)["battery_id"]), -1
        )]
    allowed = {_key_tuple(row) for row in boundary["finalized_cycle_keys"]}
    return [row for row in rows if _key_tuple(_cycle_key(row)) in allowed]
