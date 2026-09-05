"""Immutable handoff for one shared RUL/Survival generation snapshot."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path


SCHEMA_VERSION = "shared-generation-receipt-v1"


def _sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def receipt_path(root, state_id, generation):
    return Path(root) / "shared_generation_receipts" / state_id / f"{generation}.json"


def build_receipt(state_manifest_path, generation, *, root):
    root = Path(root)
    state_manifest_path = Path(state_manifest_path)
    state = json.loads(state_manifest_path.read_text())
    state_id = state["state_id"]
    benchmark = root / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    if not benchmark.is_file():
        raise ValueError("fixed offline benchmark is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": str(generation),
        "state_id": state_id,
        "state_manifest_path": str(state_manifest_path),
        "state_manifest_sha256": _sha256(state_manifest_path),
        "finalized_cycle_boundary_ref": state["finalized_cycle_boundary_ref"],
        "finalized_cycle_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
        "feature_contract_version": state["feature_contract_version"],
        "arrival_manifest_fingerprint": state["arrival_manifest_fingerprint"],
        "cutoff_metadata": state["cutoff_metadata"],
        "cohort_checksums": {
            name: sha256("\n".join(state.get(name, ())).encode()).hexdigest()
            for name in ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")
        },
        "training_features_path": str(root / "historical_features" / state_id),
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


def read_receipt(path):
    receipt = json.loads(Path(path).read_text())
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shared generation receipt")
    if _sha256(receipt["state_manifest_path"]) != receipt["state_manifest_sha256"]:
        raise ValueError("stream state manifest changed after shared generation planning")
    if _sha256(receipt["benchmark_path"]) != receipt["benchmark_sha256"]:
        raise ValueError("fixed offline benchmark changed after shared generation planning")
    return receipt


def _artifact_dir(receipt, family):
    root = Path(receipt["training_features_path"]).parents[1]
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
    args = parser.parse_args()
    if args.state_manifest:
        if not args.generation:
            raise SystemExit("--generation is required with --state-manifest")
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
