"""Fit immutable, lineage-safe RUL candidate generations."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .continuous_arrival import SNAPSHOT_THRESHOLDS, eligible_training_batteries, model_fingerprint, next_snapshot
    from .generation_snapshots import SEMANTICS_VERSION, build_generation_plan
    from .progressive_arrival import SEMANTICS_VERSION as PROGRESSIVE_SEMANTICS_VERSION, build_plan as build_progressive_plan
    from .feature_contract import RUL_FEATURES
    from .rul_predictions import constrain_prediction_row
except ImportError:
    from continuous_arrival import SNAPSHOT_THRESHOLDS, eligible_training_batteries, model_fingerprint, next_snapshot
    from generation_snapshots import SEMANTICS_VERSION, build_generation_plan
    from progressive_arrival import SEMANTICS_VERSION as PROGRESSIVE_SEMANTICS_VERSION, build_plan as build_progressive_plan
    from feature_contract import RUL_FEATURES
    from rul_predictions import constrain_prediction_row

ROOT = Path("data/processed/matr")
TARGET = "rul_cycles"
SPLIT_VERSION = "lineage-split-42"
FAMILY_ORDER = ("ridge", "random_forest", "xgboost", "mlp")
SELECTION_POLICY = {"metric": "validation_mae", "tie_breakers": ("validation_rmse", "family_order")}
MODEL_FAMILIES = {
    "ridge": (
        {"id": "alpha-0.1", "alpha": 0.1},
        {"id": "alpha-10", "alpha": 10.0},
    ),
    "random_forest": (
        {"id": "depth-8-leaf-1", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 1, "n_jobs": 4, "random_state": 42},
        {"id": "unbounded-leaf-2", "n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "n_jobs": 4, "random_state": 42},
    ),
    "xgboost": (
        {"id": "trees-200-depth-6", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "n_jobs": 4, "random_state": 42},
        {"id": "trees-300-depth-4", "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "n_jobs": 4, "random_state": 42},
    ),
    "mlp": (
        {"id": "hidden-64-alpha-0.0001", "hidden_layer_sizes": (64,), "alpha": 0.0001, "random_state": 42, "max_iter": 500, "early_stopping": True},
        {"id": "hidden-128-64-alpha-0.001", "hidden_layer_sizes": (128, 64), "alpha": 0.001, "random_state": 42, "max_iter": 500, "early_stopping": True},
    ),
}
FEATURE_VERSION = "degradation-features:" + sha256(",".join(RUL_FEATURES).encode()).hexdigest()


def metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)), "rmse": float(mean_squared_error(y, prediction) ** .5), "r2": float(r2_score(y, prediction))}


def _model(family, config, *, native_threads=None):
    params = {key: value for key, value in config.items() if key != "id"}
    if native_threads is not None and "n_jobs" in params:
        params["n_jobs"] = native_threads
    if family == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(**params))
    if family == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**params))
    if family == "xgboost":
        from xgboost import XGBRegressor
        return make_pipeline(SimpleImputer(strategy="median"), XGBRegressor(**params))
    if family == "mlp":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(**params))
    raise ValueError(f"unknown model family: {family}")


def select_validation_winner(family_results):
    """Return the deterministic validation winner without access to test data."""
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    family, result = min(
        family_results.items(),
        key=lambda item: (item[1]["validation"]["mae"], item[1]["validation"]["rmse"], family_rank[item[0]]),
    )
    return {"family": family, **result}


def select_model(train_matrix, train_labels, validation_matrix, validation_labels, *, native_threads=None):
    """Fit and select from training/validation data only."""
    family_results, fitted = {}, {}
    for family in FAMILY_ORDER:
        candidates = []
        for config in MODEL_FAMILIES[family]:
            model = _model(family, config, native_threads=native_threads)
            model.fit(train_matrix, train_labels)
            validation = metrics(validation_labels, np.maximum(model.predict(validation_matrix), 0))
            candidates.append({"config_id": config["id"], "config": config, "validation": validation, "model": model})
        best = min(candidates, key=lambda item: (item["validation"]["mae"], item["validation"]["rmse"], item["config_id"]))
        family_results[family] = {
            "config_id": best["config_id"],
            "config": best["config"],
            "validation": best["validation"],
            "configuration_results": [{key: value for key, value in item.items() if key != "model"} for item in candidates],
        }
        fitted[family] = best["model"]
    winner = select_validation_winner(family_results)
    return fitted[winner["family"]], winner, family_results


def fixed_benchmark_ids(manifest):
    """Keep validation/test lineage membership fixed while excluding invalid labels."""
    return {split: {row["battery_id"] for row in manifest if row["split"] == split and row["valid_eol_label"]} for split in ("validation", "test")}


def _manifest_fingerprint(manifest):
    fingerprints = {row["schedule_fingerprint"] for row in manifest}
    if len(fingerprints) != 1:
        raise ValueError("arrival manifest must contain one schedule fingerprint")
    return fingerprints.pop()


def _published_counts(output_root):
    counts = set()
    for path in (Path(output_root) / "model_generations").glob("*/candidate_model_evaluation.parquet"):
        row = pq.read_table(path).to_pylist()[0]
        counts.add(int(json.loads(row["training_metadata_json"])["training_battery_count"]))
    return counts


def generation_plan(manifest, lifecycle_events, published_counts=None, *, output_root=None, eligible_battery_ids=None, stream_state_id=None):
    """Return the next deterministic candidate, or None when no snapshot is due."""
    published_counts = set(published_counts or ())
    if output_root:
        published_counts.update(_published_counts(output_root))
    eligible = set(eligible_battery_ids) if eligible_battery_ids is not None else eligible_training_batteries(manifest, lifecycle_events)
    ordered = [row["battery_id"] for row in sorted(manifest, key=lambda row: row["arrival_rank"]) if row["battery_id"] in eligible]
    snapshot = next_snapshot(ordered, published_counts)
    if snapshot is None:
        return None
    threshold, battery_ids = snapshot
    feature_hash = FEATURE_VERSION.rsplit(":", 1)[1]
    manifest_fingerprint = _manifest_fingerprint(manifest)
    fingerprint = model_fingerprint(
        battery_ids,
        manifest_fingerprint=manifest_fingerprint,
        split_version=SPLIT_VERSION,
        feature_version=FEATURE_VERSION,
        model_config={"model_families": MODEL_FAMILIES, "selection_policy": SELECTION_POLICY, "feature_list_hash": feature_hash},
    )
    artifact_dir = Path(output_root or ROOT) / "model_generations" / fingerprint
    if (artifact_dir / "candidate_model_evaluation.parquet").exists():
        return None
    generation = f"1.{SNAPSHOT_THRESHOLDS.index(threshold)}"
    metadata = {"generation": generation, "model_fingerprint": fingerprint, "training_battery_count": threshold, "snapshot_lineage_checksum": sha256("\n".join(battery_ids).encode()).hexdigest(), "arrival_manifest_fingerprint": manifest_fingerprint, "split_version": SPLIT_VERSION, "feature_version": FEATURE_VERSION, "feature_list_hash": feature_hash, "model_families": MODEL_FAMILIES, "selection_policy": SELECTION_POLICY, **({"stream_state_id": stream_state_id} if stream_state_id else {})}
    return {"battery_ids": battery_ids, "training_battery_count": threshold, "fingerprint": fingerprint, "generation": generation, "model_version": f"matr-rul-model-{generation}-{fingerprint[:12]}", "metadata": metadata, "artifact_dir": artifact_dir}


def shared_generation_plan(manifest, state_manifest, generation, *, output_root=ROOT):
    """Create the RUL half of a shared RUL/Survival stream-state generation."""
    model_config = {"model_families": MODEL_FAMILIES, "selection_policy": SELECTION_POLICY}
    progressive = state_manifest.get("generation_semantics_version") == PROGRESSIVE_SEMANTICS_VERSION
    plan = (build_progressive_plan(generation, manifest, state_manifest, model_config=model_config,
        feature_version=FEATURE_VERSION, artifact_root=output_root)
        if progressive else build_generation_plan(generation, manifest, state_manifest,
        model_config=model_config, feature_version=FEATURE_VERSION, artifact_root=output_root))
    fingerprint = plan["fingerprint"]
    metadata = {
        "generation": generation, "model_fingerprint": fingerprint,
        "training_battery_count": len(plan["arrived_train_battery_ids"]),
        "supervised_training_battery_count": len(plan["observed_eol_train_battery_ids"]),
        "snapshot_lineage_checksum": plan["cohort_checksums"]["arrived_train_battery_ids"],
        "arrival_manifest_fingerprint": state_manifest["arrival_manifest_fingerprint"],
        "split_version": SPLIT_VERSION, "feature_version": FEATURE_VERSION,
        "feature_list_hash": FEATURE_VERSION.rsplit(":", 1)[1], "model_families": MODEL_FAMILIES,
        "selection_policy": SELECTION_POLICY, "stream_state_id": plan["snapshot_id"],
        "generation_semantics_version": plan["generation_semantics_version"],
        "finalized_cycle_boundary_fingerprint": state_manifest["finalized_cycle_boundary_fingerprint"],
        "cohort_checksums": plan["cohort_checksums"],
    }
    if progressive:
        metadata["record_class"] = state_manifest["record_class"]
        metadata["as_of_cutoff"] = plan["cutoff"].isoformat()
        metadata["progressive_arrival_registry"] = state_manifest["progressive_arrival_registry"]
    return {**plan, "battery_ids": plan["observed_eol_train_battery_ids"],
            "model_version": f"matr-rul-model-{generation}-{fingerprint[:12]}", "metadata": metadata,
            "artifact_dir": Path(output_root) / "model_generations" / fingerprint,
            **({"scheduled_manifest_path": Path(output_root) / state_manifest["scheduled_arrival_manifest_ref"]} if progressive else {})}


def train_generation(plan, *, root=ROOT, evaluated_at=None):
    """Train exactly one immutable candidate artifact directory."""
    if plan is None:
        return None
    root, artifact_dir = Path(root), plan["artifact_dir"]
    evaluation_path = artifact_dir / "candidate_model_evaluation.parquet"
    if evaluation_path.exists():
        return artifact_dir
    evaluated_at = evaluated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    table = ds.dataset(root / "degradation_features", format="parquet").to_table().to_pydict()
    manifest = pq.read_table(root / "arrival_manifest.parquet").to_pylist()
    battery = np.asarray(table["battery_id"])
    labels = np.asarray([np.nan if value is None else value for value in table[TARGET]], float)
    matrix = np.asarray([[np.nan if value is None else value for value in table[column]] for column in RUL_FEATURES], float).T
    matrix[~np.isfinite(matrix)] = np.nan
    benchmark = fixed_benchmark_ids(manifest)
    valid_label = np.isfinite(labels)
    indexes = {"train": valid_label & np.isin(battery, plan["battery_ids"]), "validation": valid_label & np.isin(battery, list(benchmark["validation"])), "test": valid_label & np.isin(battery, list(benchmark["test"]))}
    if not all(index.any() for index in indexes.values()):
        raise ValueError("candidate requires valid training, validation, and test rows")
    train_matrix, train_labels = matrix[indexes["train"]], labels[indexes["train"]]
    if plan.get("training_features_path"):
        state_table = ds.dataset(plan["training_features_path"], format="parquet").to_table().to_pydict()
        state_battery = np.asarray(state_table["battery_id"])
        state_labels = np.asarray([np.nan if value is None else value for value in state_table[TARGET]], float)
        state_matrix = np.asarray([[np.nan if value is None else value for value in state_table[column]] for column in RUL_FEATURES], float).T
        state_matrix[~np.isfinite(state_matrix)] = np.nan
        state_index = np.isfinite(state_labels) & np.isin(state_battery, plan["battery_ids"])
        if not state_index.any():
            raise ValueError("state-bound RUL features contain no observed-EOL training rows")
        train_matrix, train_labels = state_matrix[state_index], state_labels[state_index]
    model, winner, family_results = select_model(
        train_matrix, train_labels, matrix[indexes["validation"]], labels[indexes["validation"]],
        native_threads=plan.get("native_threads"),
    )
    raw_predictions = model.predict(matrix)
    score = {
        "train": metrics(labels[indexes["train"]], np.maximum(raw_predictions[indexes["train"]], 0)),
        "validation": winner["validation"],
        "family_validation": family_results,
        "selection": {"selected_family": winner["family"], "selected_config_id": winner["config_id"], **SELECTION_POLICY},
        "test": metrics(labels[indexes["test"]], np.maximum(raw_predictions[indexes["test"]], 0)),
    }
    score["generalization_gap"] = {key: score["validation"][key] - score["train"][key] for key in score["train"]}
    cycles, eol = np.asarray(table["cycle_index"]), np.asarray(table["eol_cycle"])
    stages = np.select([cycles < eol * .33, cycles < eol * .67], ["early", "mid"], default="late")
    test_stages, test_predictions = stages[indexes["test"]], np.maximum(raw_predictions[indexes["test"]], 0)
    score["lifecycle_stage_mae"] = {stage: float(mean_absolute_error(labels[indexes["test"]][test_stages == stage], test_predictions[test_stages == stage])) for stage in ("early", "mid", "late") if (test_stages == stage).any()}
    split_by_battery = {row["battery_id"]: row["split"] for row in manifest}
    predictions = [constrain_prediction_row({"model_version": plan["model_version"], "dataset": table["dataset"][i], "battery_id": table["battery_id"][i], "cycle_index": table["cycle_index"][i], "predicted_rul_cycles": float(raw_predictions[i]), "prediction_created_at": evaluated_at, "split": split_by_battery[table["battery_id"][i]]}) for i in range(len(raw_predictions))]
    metadata = {**plan["metadata"], "selected_family": winner["family"], "selected_config_id": winner["config_id"], "state_train_row_count": int(len(train_labels)), **({"native_threads": plan["native_threads"]} if plan.get("native_threads") is not None else {}), **{f"{split}_row_count": int(index.sum()) for split, index in indexes.items()}}
    evaluation = {"model_version": plan["model_version"], "model_name": winner["family"], "dataset": "MATR", "status": "candidate", "evaluated_at": evaluated_at, "model_fingerprint": plan["fingerprint"], "generation": plan["generation"], "metrics_json": json.dumps(score, sort_keys=True), "training_metadata_json": json.dumps(metadata, sort_keys=True)}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "selected_model.joblib")
    pq.write_table(pa.Table.from_pylist(predictions), artifact_dir / "candidate_predictions.parquet")
    pq.write_table(pa.Table.from_pylist([evaluation]), evaluation_path)
    return artifact_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true", help="daily no-op-safe eligibility mode")
    parser.add_argument("--manifest", type=Path, default=ROOT / "arrival_manifest.parquet")
    parser.add_argument("--lifecycle-events", type=Path, default=ROOT / "replay_lifecycle_state")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--generation")
    parser.add_argument("--training-features", type=Path)
    parser.add_argument("--native-threads", type=int, help="limit native model workers for a bounded Airflow task")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = pq.read_table(args.manifest).to_pylist()
    lifecycle_events = pq.read_table(args.lifecycle_events).to_pylist()
    if bool(args.state_manifest) != bool(args.generation):
        raise SystemExit("--state-manifest and --generation must be supplied together")
    plan = (shared_generation_plan(manifest, json.loads(args.state_manifest.read_text()), args.generation, output_root=args.root)
            if args.state_manifest else generation_plan(manifest, lifecycle_events, output_root=args.root))
    if args.training_features:
        plan["training_features_path"] = args.training_features
    if args.native_threads is not None:
        plan["native_threads"] = args.native_threads
    artifact = train_generation(plan, root=args.root)
    if artifact and not args.state_manifest:
        (args.root / "latest_candidate_generation.txt").write_text(str(artifact))
    print(f"Published candidate artifact {artifact}" if artifact else "No new eligible battery snapshot.")


if __name__ == "__main__":
    main()
