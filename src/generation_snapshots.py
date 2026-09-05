"""Immutable shared RUL/Survival training snapshots."""
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    from .continuous_arrival import model_fingerprint
    from .feature_contract import RUL_FEATURES
    from .stream_state import build_compact_finalized_cycle_boundary, create_stream_state_manifest
except ImportError:
    from continuous_arrival import model_fingerprint
    from feature_contract import RUL_FEATURES
    from stream_state import build_compact_finalized_cycle_boundary, create_stream_state_manifest


SEMANTICS_VERSION = "shared-stream-state-v2"
GENERATION_SNAPSHOTS = (
    ("1.0", "2022-12-02T00:00:00+00:00"),
    ("1.1", "2023-05-29T00:00:00+00:00"),
    ("1.2", "2025-08-14T00:00:00+00:00"),
    ("1.3", "2026-04-08T00:00:00+00:00"),
)


def _time(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _checksum(values):
    return sha256("\n".join(values).encode()).hexdigest()


def snapshot_definition(generation):
    for name, cutoff in GENERATION_SNAPSHOTS:
        if name == generation:
            return name, _time(cutoff)
    raise ValueError(f"unknown generation snapshot: {generation}")


def cohort_at_cutoff(manifest, lifecycle_events, cutoff):
    """Return ordered arrived, observed-EOL, and censored train cohorts as of cutoff."""
    cutoff = _time(cutoff)
    observed = {event["battery_id"] for event in lifecycle_events if event.get("event_type") == "eol_observed" or event.get("eol_observed")}
    arrived = []
    observed_eol = []
    for row in sorted((row for row in manifest if row["split"] == "train"), key=lambda row: row["arrival_rank"]):
        start = _time(row["start_time"])
        if start > cutoff:
            continue
        arrived.append(row["battery_id"])
        eol_time = start
        eol_cycle = row.get("eol_cycle")
        first = int(row.get("first_source_cycle", 1))
        if eol_cycle is not None:
            eol_time = start + timedelta(days=int(eol_cycle) - first)
        if row.get("valid_eol_label") and row["battery_id"] in observed and eol_time <= cutoff:
            observed_eol.append(row["battery_id"])
    observed_set = set(observed_eol)
    return {"arrived_train_battery_ids": arrived, "observed_eol_train_battery_ids": observed_eol,
            "censored_train_battery_ids": [battery_id for battery_id in arrived if battery_id not in observed_set]}


def build_generation_plan(generation, manifest, state_manifest, *, model_config, feature_version, artifact_root,
                          cutoff=None, semantics_version=SEMANTICS_VERSION):
    """Create the one family-neutral plan consumed by both trainers."""
    _, default_cutoff = snapshot_definition(generation)
    recorded_cutoff = (state_manifest.get("cutoff_metadata") or {}).get("replay_cutoff")
    cutoff = _time(cutoff) if cutoff is not None else (_time(recorded_cutoff) if recorded_cutoff else default_cutoff)
    if recorded_cutoff and _time(recorded_cutoff) != cutoff:
        raise ValueError("state replay cutoff does not match generation snapshot")
    cohort = {key: list(state_manifest[key]) for key in ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")}
    if not cohort["arrived_train_battery_ids"] or set(cohort["observed_eol_train_battery_ids"]) - set(cohort["arrived_train_battery_ids"]):
        raise ValueError("invalid shared generation cohort")
    manifest_fingerprint = next(iter({row["schedule_fingerprint"] for row in manifest}))
    cohort_checksums = {key: _checksum(values) for key, values in cohort.items()}
    fingerprint = model_fingerprint(
        cohort["arrived_train_battery_ids"], manifest_fingerprint=manifest_fingerprint,
        split_version="lineage-split-42", feature_version=feature_version,
        model_config={"generation": generation, "semantics_version": semantics_version,
                      "state_id": state_manifest["state_id"], "boundary_fingerprint": state_manifest["finalized_cycle_boundary_fingerprint"],
                      "cohort_checksums": cohort_checksums, "model_config": model_config},
    )
    return {"generation": generation, "cutoff": cutoff, "snapshot_id": state_manifest["state_id"],
            "state_manifest": state_manifest, "fingerprint": fingerprint, "cohort_checksums": cohort_checksums,
            "training_battery_count": len(cohort["arrived_train_battery_ids"]), **cohort,
            "artifact_root": Path(artifact_root), "generation_semantics_version": semantics_version}


def reconstruct_snapshot(generation, *, root=Path("data/processed/matr")):
    """Persist the deterministic boundary and state manifest for one historical cutoff."""
    root = Path(root)
    _, cutoff = snapshot_definition(generation)
    manifest = pq.read_table(root / "arrival_manifest.parquet").to_pylist()
    lifecycle = pq.read_table(root / "replay_lifecycle_state").to_pylist()
    cohort = cohort_at_cutoff(manifest, lifecycle, cutoff)
    by_id = {row["battery_id"]: row for row in manifest}
    canonical_keys = ds.dataset(root / "cycle_summary", format="parquet").to_table(columns=["dataset", "battery_id", "cycle_index"]).to_pylist()
    keys = []
    for row in canonical_keys:
        source = by_id[row["battery_id"]]
        start = _time(source["start_time"])
        available = int((cutoff - start).days) + int(source.get("first_source_cycle", 1))
        if start <= cutoff and int(row["cycle_index"]) <= min(int(source["last_source_cycle"]), available):
            keys.append({"dataset": row["dataset"], "battery_id": row["battery_id"], "cycle_index": int(row["cycle_index"]), "replay_sequence": 0})
    boundary = build_compact_finalized_cycle_boundary(keys, canonical_cycle_keys=canonical_keys, canonical_fingerprint="matr-canonical-v1",
        arrival_manifest_fingerprint=next(iter({row["schedule_fingerprint"] for row in manifest})),
        feature_contract_version="degradation-features:" + sha256(",".join(RUL_FEATURES).encode()).hexdigest())
    provisional = create_stream_state_manifest(boundary, boundary_ref="pending", eligible_completed_training_batteries=cohort["observed_eol_train_battery_ids"], cutoff_metadata={"replay_cutoff": cutoff.isoformat(), "generation": generation}, kafka_offsets={})
    state_id = provisional["state_id"]
    boundary_path = root / "finalized_cycle_boundary" / state_id / "boundary.json"
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_path.write_text(json.dumps(boundary, sort_keys=True, indent=2))
    state = create_stream_state_manifest(boundary, boundary_ref=str(boundary_path.relative_to(root)), eligible_completed_training_batteries=cohort["observed_eol_train_battery_ids"], cutoff_metadata={"replay_cutoff": cutoff.isoformat(), "generation": generation}, kafka_offsets={})
    state = {**state, **cohort, "generation": generation, "generation_semantics_version": SEMANTICS_VERSION}
    state_path = root / "stream_state" / state_id / "manifest.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True, indent=2))
    return state_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct immutable shared generation snapshots.")
    parser.add_argument("--generation", choices=[generation for generation, _ in GENERATION_SNAPSHOTS])
    parser.add_argument("--root", type=Path, default=Path("data/processed/matr"))
    parser.add_argument("--latest", type=Path, help="write the selected immutable state-manifest path for orchestration")
    args = parser.parse_args()
    generations = [args.generation] if args.generation else [generation for generation, _ in GENERATION_SNAPSHOTS]
    paths = [reconstruct_snapshot(generation, root=args.root) for generation in generations]
    if args.latest:
        args.latest.parent.mkdir(parents=True, exist_ok=True)
        args.latest.write_text(str(paths[-1]))
    for path in paths:
        print(path)
