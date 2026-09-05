"""Immutable handoff for one shared RUL/Survival generation snapshot."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

try:
    from .shared_features import (
        generation_id_for, load_shared_feature_rows, selected_rows_digest,
        validate_feature_outlet, validate_shared_features,
    )
    from .stream_state import validate_stream_state_manifest
except ImportError:
    from shared_features import (
        generation_id_for, load_shared_feature_rows, selected_rows_digest,
        validate_feature_outlet, validate_shared_features,
    )
    from stream_state import validate_stream_state_manifest

SCHEMA_VERSION = "shared-generation-receipt-v3"
LEGACY_SCHEMA_VERSION = "shared-generation-receipt-v2"


def _sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _cohort_checksums(state):
    return {
        name: sha256("\n".join(state.get(name, ())).encode()).hexdigest()
        for name in ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")
    }


def _cohort_checksum(state):
    cohort = {name: state.get(name, ()) for name in (
        "arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids"
    )}
    return sha256(json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def receipt_path(root, state_id, generation):
    return Path(root) / "shared_generation_receipts" / state_id / f"{generation}.v3.json"


def build_receipt(state_manifest_path, generation, *, root):
    root = Path(root)
    state_manifest_path = Path(state_manifest_path)
    state = json.loads(state_manifest_path.read_text())
    state_id = state["state_id"]
    benchmark = root / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    if not benchmark.is_file():
        raise ValueError("fixed offline benchmark is required")
    state = validate_state_manifest_path(state_manifest_path, root=root)
    boundary_path = root / state["finalized_cycle_boundary_ref"]
    boundary = json.loads(boundary_path.read_text())
    generation_id = generation_id_for(generation)
    outlet_path = root / "shared_feature_outlet"
    validate_feature_outlet(
        outlet_path, feature_contract_version=state["feature_contract_version"],
        canonical_source_fingerprint=state["canonical_fingerprint"],
    )
    selected = load_shared_feature_rows(root, generation_id=generation_id, boundary=boundary)
    if not selected:
        raise ValueError("shared feature outlet contains no rows for the generation boundary")
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": str(generation),
        "generation_id": generation_id,
        "state_id": state_id,
        "state_manifest_path": str(state_manifest_path),
        "state_manifest_sha256": _sha256(state_manifest_path),
        "state_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
        "feature_contract_version": state["feature_contract_version"],
        "canonical_source_fingerprint": state["canonical_fingerprint"],
        "cutoff_metadata": state["cutoff_metadata"],
        "cohort_checksum": _cohort_checksum(state),
        "selected_row_count": len(selected),
        "selected_rows_sha256": selected_rows_digest(selected),
        "shared_feature_outlet_ref": str(outlet_path.relative_to(root)),
        "benchmark_path": str(benchmark),
        "benchmark_sha256": _sha256(benchmark),
    }


def write_receipt(path, receipt):
    """Atomically create an exact-content receipt; retries may only repeat it."""
    path = Path(path)
    encoded = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"immutable shared generation receipt differs: {path}")
        return path
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_text() != encoded:
            raise ValueError(f"immutable shared generation receipt differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_legacy_receipt(receipt):
    if _sha256(receipt["state_manifest_path"]) != receipt["state_manifest_sha256"]:
        raise ValueError("stream state manifest changed after shared generation planning")
    state = json.loads(Path(receipt["state_manifest_path"]).read_text())
    required = {
        "state_id": state["state_id"],
        "finalized_cycle_boundary_ref": state["finalized_cycle_boundary_ref"],
        "finalized_cycle_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
        "feature_contract_version": state["feature_contract_version"],
        "arrival_manifest_fingerprint": state["arrival_manifest_fingerprint"],
        "cutoff_metadata": state["cutoff_metadata"],
        "cohort_checksums": _cohort_checksums(state),
    }
    for field, value in required.items():
        if receipt.get(field) != value:
            raise ValueError(f"shared generation receipt {field.replace('_', ' ')} mismatch")
    root = Path(receipt["training_features_path"]).parents[1]
    try:
        validate_state_manifest_path(receipt["state_manifest_path"], root=root)
    except (KeyError, ValueError) as error:
        raise ValueError(f"shared generation receipt boundary validation failed: {error}") from error
    if _sha256(receipt["benchmark_path"]) != receipt["benchmark_sha256"]:
        raise ValueError("fixed offline benchmark changed after shared generation planning")
    validate_shared_features(
        receipt["training_features_path"],
        state_manifest_path=receipt["state_manifest_path"],
        generation=receipt["generation"],
        expected=receipt["training_features"],
    )
    return receipt


def read_receipt(path):
    path = Path(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") == LEGACY_SCHEMA_VERSION:
        return _read_legacy_receipt(receipt)
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shared generation receipt")
    if _sha256(receipt["state_manifest_path"]) != receipt["state_manifest_sha256"]:
        raise ValueError("stream state manifest changed after shared generation planning")
    root = path.parents[2]
    state = validate_state_manifest_path(receipt["state_manifest_path"], root=root)
    required = {
        "generation_id": generation_id_for(receipt["generation"]),
        "state_id": state["state_id"],
        "state_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
        "feature_contract_version": state["feature_contract_version"],
        "canonical_source_fingerprint": state["canonical_fingerprint"],
        "cutoff_metadata": state["cutoff_metadata"],
        "cohort_checksum": _cohort_checksum(state),
        "shared_feature_outlet_ref": "shared_feature_outlet",
    }
    for field, value in required.items():
        if receipt.get(field) != value:
            raise ValueError(f"shared generation receipt {field.replace('_', ' ')} mismatch")
    if _sha256(receipt["benchmark_path"]) != receipt["benchmark_sha256"]:
        raise ValueError("fixed offline benchmark changed after shared generation planning")
    boundary = json.loads((root / state["finalized_cycle_boundary_ref"]).read_text())
    rows = load_shared_feature_rows(root, generation_id=receipt["generation_id"], boundary=boundary)
    if receipt.get("selected_row_count") != len(rows):
        raise ValueError("shared generation receipt selected row count mismatch")
    if receipt.get("selected_rows_sha256") != selected_rows_digest(rows):
        raise ValueError("shared generation receipt selected rows digest mismatch")
    return receipt


def load_receipt_feature_rows(receipt, *, root):
    """Load the already-validated cumulative outlet selection for receipt v3."""
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("legacy receipts use their immutable historical feature snapshot")
    state = json.loads(Path(receipt["state_manifest_path"]).read_text())
    boundary = json.loads((Path(root) / state["finalized_cycle_boundary_ref"]).read_text())
    rows = load_shared_feature_rows(Path(root), generation_id=receipt["generation_id"], boundary=boundary)
    if len(rows) != receipt["selected_row_count"] or selected_rows_digest(rows) != receipt["selected_rows_sha256"]:
        raise ValueError("shared generation receipt selected rows changed")
    return rows


def validate_state_manifest_path(path, *, root, require_streaming=False):
    """Validate an authoritative state and optionally require Kafka lineage."""
    path, root = Path(path), Path(root)
    state = json.loads(path.read_text())
    boundary_path = Path(state["finalized_cycle_boundary_ref"])
    if not boundary_path.is_absolute():
        boundary_path = root / boundary_path
    boundary = json.loads(boundary_path.read_text())
    validate_stream_state_manifest(
        state, boundary,
        expected_canonical_fingerprint=state["canonical_fingerprint"],
        expected_arrival_manifest_fingerprint=state["arrival_manifest_fingerprint"],
        expected_feature_contract_version=state["feature_contract_version"],
    )
    if require_streaming and not state.get("kafka_offsets"):
        raise ValueError(
            "canonical continuous training requires a Streaming-issued state with Kafka offsets; "
            "use generation_snapshots.py only for explicit offline/backfill work"
        )
    shared_fields = ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")
    if require_streaming and not all(field in state for field in shared_fields):
        raise ValueError(
            "canonical continuous training requires a Streaming-issued shared training cohort; "
            "use generation_snapshots.py only for explicit offline/backfill work"
        )
    if require_streaming and not state.get("cutoff_metadata", {}).get("replay_cutoff"):
        raise ValueError("canonical continuous training requires the Streaming replay cutoff")
    return state


def _artifact_dir(receipt, family):
    root = (Path(receipt["training_features_path"]).parents[1]
            if receipt.get("schema_version") == LEGACY_SCHEMA_VERSION
            else Path(receipt["state_manifest_path"]).parents[2])
    import pyarrow.parquet as pq

    manifest = pq.read_table(root / "arrival_manifest.parquet").to_pylist()
    state = json.loads(Path(receipt["state_manifest_path"]).read_text())
    if family == "rul":
        try:
            from .train_matr_models import shared_generation_plan
        except ImportError:
            from train_matr_models import shared_generation_plan
        return shared_generation_plan(manifest, state, receipt["generation"], output_root=root)["artifact_dir"]
    if family == "survival":
        try:
            from .survival_models import survival_generation_plan
        except ImportError:
            from survival_models import survival_generation_plan
        return survival_generation_plan(manifest, state, receipt["generation"], root=root)["artifact_dir"]
    raise ValueError(f"unknown generation family: {family}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or inspect a shared generation receipt.")
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--generation")
    parser.add_argument("--root", type=Path, default=Path("data/processed/matr"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--field")
    parser.add_argument("--artifact-dir", choices=("rul", "survival"))
    parser.add_argument("--validate-state-only", action="store_true")
    parser.add_argument("--require-streaming", action="store_true")
    args = parser.parse_args()
    if args.state_manifest:
        if not args.generation:
            raise SystemExit("--generation is required with --state-manifest")
        state = validate_state_manifest_path(args.state_manifest, root=args.root, require_streaming=args.require_streaming)
        if args.validate_state_only:
            print(state[args.field] if args.field else args.state_manifest)
            raise SystemExit(0)
        receipt = build_receipt(args.state_manifest, args.generation, root=args.root)
        print(write_receipt(receipt_path(args.root, receipt["state_id"], args.generation), receipt))
    elif args.receipt:
        receipt = read_receipt(args.receipt)
        if args.field:
            print(receipt[args.field])
        elif args.artifact_dir:
            print(_artifact_dir(receipt, args.artifact_dir))
        else:
            print(json.dumps(receipt, sort_keys=True))
    else:
        raise SystemExit("provide --state-manifest or --receipt")
