"""Immutable, EOL-independent progressive train-arrival snapshots."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    from .feature_contract import RUL_FEATURES
    from .generation_snapshots import build_generation_plan, cohort_at_cutoff
    from .stream_state import build_finalized_cycle_boundary, create_stream_state_manifest
except ImportError:
    from feature_contract import RUL_FEATURES
    from generation_snapshots import build_generation_plan, cohort_at_cutoff
    from stream_state import build_finalized_cycle_boundary, create_stream_state_manifest


SEMANTICS_VERSION = "shared-progressive-arrival-v3"
RECORD_CLASS = "canonical_shared_progressive_arrival_v3"
ARRIVAL_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
ARRIVAL_GAP_DAYS = 60
GENERATION_ARRIVAL_COUNTS = {"1.0": 26, "1.1": 51, "1.2": 76, "1.3": 94}


def _time(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule_manifest(manifest):
    """Assign train arrivals from rank and cadence only; labels are never read."""
    train = sorted((row for row in manifest if row["split"] == "train"), key=lambda row: row["arrival_rank"])
    if len(train) < max(GENERATION_ARRIVAL_COUNTS.values()):
        raise ValueError("train manifest does not contain every progressive generation cohort")
    arrivals = [
        {"battery_id": row["battery_id"], "train_arrival_rank": rank + 1,
         "arrival_time": (ARRIVAL_EPOCH + timedelta(days=ARRIVAL_GAP_DAYS * rank)).isoformat()}
        for rank, row in enumerate(train)
    ]
    registry = {
        "semantics_version": SEMANTICS_VERSION, "arrival_epoch": ARRIVAL_EPOCH.isoformat(),
        "arrival_gap_days": ARRIVAL_GAP_DAYS, "generation_arrival_counts": GENERATION_ARRIVAL_COUNTS,
        "train_arrivals": arrivals,
    }
    fingerprint = _digest(registry)
    by_battery = {row["battery_id"]: row["arrival_time"] for row in arrivals}
    scheduled = [{**row, "start_time": by_battery.get(row["battery_id"], row["start_time"]),
                  "schedule_fingerprint": fingerprint} for row in manifest]
    return scheduled, {**registry, "schedule_fingerprint": fingerprint}


def snapshot_cutoff(generation, scheduled_manifest):
    try:
        count = GENERATION_ARRIVAL_COUNTS[generation]
    except KeyError as error:
        raise ValueError(f"unknown progressive generation {generation}") from error
    train = sorted((row for row in scheduled_manifest if row["split"] == "train"), key=lambda row: row["arrival_rank"])
    return _time(train[count - 1]["start_time"])


def dry_run(manifest, lifecycle_events):
    """Report replay outcomes after schedule-only snapshot construction."""
    scheduled, registry = schedule_manifest(manifest)
    train_count = sum(row["split"] == "train" for row in scheduled)
    result = {}
    for generation, arrived_count in GENERATION_ARRIVAL_COUNTS.items():
        cutoff = snapshot_cutoff(generation, scheduled)
        cohort = cohort_at_cutoff(scheduled, lifecycle_events, cutoff)
        result[generation] = {
            "cutoff": cutoff.isoformat(), "arrived_train_battery_count": len(cohort["arrived_train_battery_ids"]),
            "not_yet_arrived_train_battery_count": train_count - len(cohort["arrived_train_battery_ids"]),
            "observed_eol_train_battery_count": len(cohort["observed_eol_train_battery_ids"]),
            "censored_train_battery_count": len(cohort["censored_train_battery_ids"]),
        }
        if len(cohort["arrived_train_battery_ids"]) != arrived_count:
            raise ValueError("snapshot arrival count does not match generation contract")
    return registry, result


def reconstruct_snapshot(generation, *, root=Path("data/processed/matr")):
    """Persist one state boundary from the predeclared progressive schedule."""
    root = Path(root)
    manifest = pq.read_table(root / "arrival_manifest.parquet").to_pylist()
    lifecycle = pq.read_table(root / "replay_lifecycle_state").to_pylist()
    scheduled, registry = schedule_manifest(manifest)
    schedule_root = root / "progressive_arrival_v3"
    schedule_root.mkdir(parents=True, exist_ok=True)
    scheduled_path = schedule_root / "arrival_manifest.parquet"
    if scheduled_path.exists():
        existing = pq.read_table(scheduled_path).to_pylist()
        if existing != scheduled:
            raise ValueError("progressive arrival manifest already exists with different content")
    else:
        pq.write_table(pa.Table.from_pylist(scheduled), scheduled_path)
    cutoff = snapshot_cutoff(generation, scheduled)
    cohort = cohort_at_cutoff(scheduled, lifecycle, cutoff)
    expected = GENERATION_ARRIVAL_COUNTS[generation]
    if len(cohort["arrived_train_battery_ids"]) != expected:
        raise ValueError("progressive arrival cohort is not exact")
    by_id = {row["battery_id"]: row for row in scheduled}
    arrived = set(cohort["arrived_train_battery_ids"])
    keys = []
    for row in ds.dataset(root / "cycle_summary", format="parquet").to_table(columns=["dataset", "battery_id", "cycle_index"]).to_pylist():
        source = by_id[row["battery_id"]]
        if source["battery_id"] not in arrived:
            continue
        available = int((cutoff - _time(source["start_time"])).days) + int(source.get("first_source_cycle", 1))
        if int(row["cycle_index"]) <= min(int(source["last_source_cycle"]), available):
            keys.append({"dataset": row["dataset"], "battery_id": row["battery_id"], "cycle_index": int(row["cycle_index"])})
    feature_contract = "degradation-features:" + sha256(",".join(RUL_FEATURES).encode()).hexdigest()
    boundary = build_finalized_cycle_boundary(keys, canonical_fingerprint="matr-canonical-v1",
        arrival_manifest_fingerprint=registry["schedule_fingerprint"], feature_contract_version=feature_contract)
    provisional = create_stream_state_manifest(boundary, boundary_ref="pending",
        eligible_completed_training_batteries=cohort["observed_eol_train_battery_ids"],
        cutoff_metadata={"replay_cutoff": cutoff.isoformat(), "generation": generation, "generation_semantics_version": SEMANTICS_VERSION}, kafka_offsets={})
    state_id = provisional["state_id"]
    boundary_path = root / "finalized_cycle_boundary" / state_id / "boundary.json"
    state_path = root / "stream_state" / state_id / "manifest.json"
    state = create_stream_state_manifest(boundary, boundary_ref=str(boundary_path.relative_to(root)),
        eligible_completed_training_batteries=cohort["observed_eol_train_battery_ids"],
        cutoff_metadata={"replay_cutoff": cutoff.isoformat(), "generation": generation, "generation_semantics_version": SEMANTICS_VERSION}, kafka_offsets={})
    state = {**state, **cohort, "generation": generation, "generation_semantics_version": SEMANTICS_VERSION,
             "record_class": RECORD_CLASS, "progressive_arrival_registry": registry,
             "scheduled_arrival_manifest_ref": str(scheduled_path.relative_to(root))}
    if boundary_path.exists() and json.loads(boundary_path.read_text()) != boundary:
        raise ValueError("progressive boundary already exists with different content")
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError("progressive state already exists with different content")
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_path.write_text(json.dumps(boundary, sort_keys=True, indent=2))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True, indent=2))
    return state_path


def build_plan(generation, manifest, state_manifest, *, model_config, feature_version, artifact_root):
    """Build a family-neutral v3 plan from the persisted progressive state."""
    cutoff = (state_manifest.get("cutoff_metadata") or {}).get("replay_cutoff")
    if not cutoff:
        raise ValueError("progressive state is missing its replay cutoff")
    return build_generation_plan(generation, manifest, state_manifest, model_config=model_config,
        feature_version=feature_version, artifact_root=artifact_root, cutoff=cutoff,
        semantics_version=SEMANTICS_VERSION)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct progressive-arrival v3 snapshots.")
    parser.add_argument("--generation", choices=GENERATION_ARRIVAL_COUNTS)
    parser.add_argument("--root", type=Path, default=Path("data/processed/matr"))
    parser.add_argument("--latest", type=Path)
    args = parser.parse_args()
    paths = [reconstruct_snapshot(args.generation, root=args.root)] if args.generation else [reconstruct_snapshot(generation, root=args.root) for generation in GENERATION_ARRIVAL_COUNTS]
    if args.latest:
        args.latest.parent.mkdir(parents=True, exist_ok=True)
        args.latest.write_text(str(paths[-1]))
    for path in paths:
        print(path)
