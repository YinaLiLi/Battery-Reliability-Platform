"""Durable publication of finalized Streaming state artifacts."""

import json
import os
from pathlib import Path

try:
    from .stream_state import build_finalized_cycle_boundary, create_stream_state_manifest
except ImportError:
    from stream_state import build_finalized_cycle_boundary, create_stream_state_manifest


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))
    os.replace(temporary, path)


def _immutable_json(path, value):
    """Create immutable JSON or reject a conflicting state identity."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"immutable state artifact differs: {path}")
        return
    _atomic_json(path, value)


def publish_state_artifacts(root, *, finalized_keys, state_rows, feature_rows, canonical_fingerprint, arrival_manifest_fingerprint, feature_contract_version, eligible_completed_training_batteries, cutoff_metadata, kafka_offsets, publish_latest=True, require_kafka_offsets=False):
    """Write immutable artifacts first, then atomically expose their manifest."""
    root = Path(root)
    boundary = build_finalized_cycle_boundary(
        finalized_keys, canonical_fingerprint=canonical_fingerprint,
        arrival_manifest_fingerprint=arrival_manifest_fingerprint,
        feature_contract_version=feature_contract_version,
    )
    provisional = create_stream_state_manifest(
        boundary, boundary_ref="pending", eligible_completed_training_batteries=eligible_completed_training_batteries,
        cutoff_metadata=cutoff_metadata, kafka_offsets=kafka_offsets,
    )
    state_id = provisional["state_id"]
    if require_kafka_offsets and not kafka_offsets:
        raise ValueError("Kafka-produced stream state requires source offsets")
    boundary_path = root / "finalized_cycle_boundary" / state_id / "boundary.json"
    _immutable_json(boundary_path, boundary)
    _immutable_json(root / "as_of_cycle_state" / state_id / "state.json", state_rows)
    _immutable_json(root / "as_of_cycle_features" / state_id / "features.json", feature_rows)
    manifest = create_stream_state_manifest(
        boundary, boundary_ref=str(boundary_path.relative_to(root)),
        eligible_completed_training_batteries=eligible_completed_training_batteries,
        cutoff_metadata=cutoff_metadata, kafka_offsets=kafka_offsets,
    )
    _immutable_json(root / "stream_state" / state_id / "manifest.json", manifest)
    if publish_latest:
        _atomic_json(root / "stream_state" / "latest.json", manifest)
    return manifest


def publish_latest_manifest(root, manifest):
    """Atomically expose a state only after required serving has completed."""
    _atomic_json(Path(root) / "stream_state" / "latest.json", manifest)
