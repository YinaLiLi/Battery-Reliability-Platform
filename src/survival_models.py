"""Independent landmark conditional-survival training for MATR."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .generation_snapshots import SEMANTICS_VERSION, build_generation_plan
    from .progressive_arrival import SEMANTICS_VERSION as PROGRESSIVE_SEMANTICS_VERSION, build_plan as build_progressive_plan
    from .shared_generation_receipt import SCHEMA_VERSION as RECEIPT_SCHEMA_VERSION, load_receipt_feature_rows, read_receipt
    from .train_matr_models import RUL_FEATURES, SPLIT_VERSION
except ImportError:
    from generation_snapshots import SEMANTICS_VERSION, build_generation_plan
    from progressive_arrival import SEMANTICS_VERSION as PROGRESSIVE_SEMANTICS_VERSION, build_plan as build_progressive_plan
    from shared_generation_receipt import SCHEMA_VERSION as RECEIPT_SCHEMA_VERSION, load_receipt_feature_rows, read_receipt
    from train_matr_models import RUL_FEATURES, SPLIT_VERSION

try:
    from .survival_contract import HORIZON_GRID, REPORT_HORIZONS, validate_prediction_rows
except ImportError:
    from survival_contract import HORIZON_GRID, REPORT_HORIZONS, validate_prediction_rows


ROOT = Path("data/processed/matr")
FAMILY_ORDER = ("cox", "random_survival_forest")
MODEL_FAMILIES = {
    "cox": ({"id": "alpha-0.1", "alpha": 0.1}, {"id": "alpha-1", "alpha": 1.0}),
    "random_survival_forest": ({"id": "trees-200-depth-8", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5, "n_jobs": 4, "random_state": 42},),
}
SELECTION_POLICY = {"metric": "validation_integrated_brier_score", "tie_breakers": ("validation_ipcw_c_index", "family_order")}
FEATURE_VERSION = "survival-landmark-features:" + sha256(",".join(RUL_FEATURES).encode()).hexdigest()
FORBIDDEN_FEATURES = {"eol_cycle", "rul_cycles", "duration_cycles", "event_observed", "terminal_cycle", "lifecycle_state"}
LANDMARK_SAMPLING = {"train_stride_cycles": 10, "retain": ("first", "stride", "final_positive_follow_up")}
IPCW_SUPPORT_POLICY = "training-censoring-survival-positive-v1"


def _as_datetime(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _cycle_at_cutoff(manifest_row, cutoff):
    start = _as_datetime(manifest_row["start_time"])
    if cutoff < start:
        return None
    elapsed = (cutoff - start).days
    return min(int(manifest_row["last_source_cycle"]), int(manifest_row.get("first_source_cycle", 1)) + elapsed)


def landmark_rows(feature_rows, manifest, lifecycle_events, cutoff, *, feature_columns=RUL_FEATURES, train_battery_ids=None, fixed_evaluation=False):
    """Make causal landmark rows as of cutoff; repeated rows remain battery-dependent."""
    forbidden = set(feature_columns) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError("future-derived features are forbidden: " + ", ".join(sorted(forbidden)))
    cutoff = _as_datetime(cutoff)
    observed = {event["battery_id"] for event in lifecycle_events if event.get("event_type") == "eol_observed" or event.get("eol_observed")}
    by_battery = {row["battery_id"]: row for row in manifest}
    allowed_train = None if train_battery_ids is None else set(train_battery_ids)
    output = []
    for row in feature_rows:
        source = by_battery.get(row["battery_id"])
        if source is None:
            continue
        if source["split"] == "train" and allowed_train is not None and row["battery_id"] not in allowed_train:
            continue
        available_cycle = int(source["last_source_cycle"]) if fixed_evaluation and source["split"] != "train" else _cycle_at_cutoff(source, cutoff)
        if available_cycle is None or int(row["cycle_index"]) > available_cycle:
            continue
        eol = source.get("eol_cycle")
        event = bool(source.get("valid_eol_label") and row["battery_id"] in observed and eol is not None and int(eol) <= available_cycle)
        terminal = int(eol) if event else available_cycle
        duration = terminal - int(row["cycle_index"])
        if duration <= 0:
            continue
        output.append({
            "dataset": row.get("dataset", source.get("dataset", "MATR")), "battery_id": row["battery_id"],
            "lineage_group_id": source["lineage_group_id"], "split": source["split"], "cycle_index": int(row["cycle_index"]),
            "duration_cycles": duration, "event_observed": event,
            **{column: row.get(column) for column in feature_columns},
        })
    return output


def training_landmark_rows(rows):
    """Retain deterministic train anchors while leaving evaluation landmarks intact."""
    first_cycle = {}
    for row in rows:
        if row["split"] == "train":
            first_cycle[row["battery_id"]] = min(first_cycle.get(row["battery_id"], row["cycle_index"]), row["cycle_index"])
    stride = LANDMARK_SAMPLING["train_stride_cycles"]
    return [
        row for row in rows
        if row["split"] != "train"
        or row["cycle_index"] == first_cycle[row["battery_id"]]
        or row["cycle_index"] % stride == 0
        or row["duration_cycles"] == 1
    ]


def select_validation_winner(family_results):
    rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    family, result = min(
        family_results.items(),
        key=lambda item: (
            item[1]["validation"].get("integrated_brier_score") is None,
            item[1]["validation"].get("integrated_brier_score", float("inf")),
            -item[1]["validation"].get("ipcw_c_index", float("-inf")), rank[item[0]],
        ),
    )
    return {"family": family, **result}


def _model(family, config, *, native_threads=None):
    params = {key: value for key, value in config.items() if key != "id"}
    if native_threads is not None and "n_jobs" in params:
        params["n_jobs"] = native_threads
    if family == "cox":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), CoxPHSurvivalAnalysis(**params))
    if family == "random_survival_forest":
        from sksurv.ensemble import RandomSurvivalForest
        return make_pipeline(SimpleImputer(strategy="median"), RandomSurvivalForest(**params))
    raise ValueError(f"unknown survival family: {family}")


def _target(rows):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray([row["event_observed"] for row in rows], bool), time=np.asarray([row["duration_cycles"] for row in rows], float))


def _matrix(rows, feature_columns):
    matrix = np.asarray([[np.nan if row.get(column) is None else row[column] for column in feature_columns] for row in rows], float)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _curve_values(model, matrix, times):
    functions = model.predict_survival_function(matrix)
    return np.asarray([[1.0 if time == 0 else float(function(time)) for time in times] for function in functions])


def _ipcw_supported_rows(train_rows, rows):
    """Restrict IPCW scoring to the observed training censoring support."""
    from sksurv.nonparametric import CensoringDistributionEstimator

    maximum = max(row["duration_cycles"] for row in train_rows)
    candidates = [row for row in rows if row["duration_cycles"] <= maximum]
    if not candidates:
        return []
    censoring = CensoringDistributionEstimator().fit(_target(train_rows))
    probabilities = censoring.predict_proba(np.asarray([row["duration_cycles"] for row in candidates], float))
    return [row for row, probability in zip(candidates, probabilities) if probability > 0.0]


def _metrics(train_rows, rows, model, feature_columns):
    from sksurv.metrics import brier_score, concordance_index_ipcw, integrated_brier_score
    supported_rows = _ipcw_supported_rows(train_rows, rows)
    if not supported_rows:
        raise ValueError("no evaluation landmarks fall within training IPCW support")
    train_y, evaluation_y = _target(train_rows), _target(supported_rows)
    matrix = _matrix(supported_rows, feature_columns)
    risk = model.predict(matrix)
    c_index = float(concordance_index_ipcw(train_y, evaluation_y, risk)[0])
    maximum = int(np.max(evaluation_y["time"]))
    times = np.asarray([time for time in HORIZON_GRID if 0 < time < maximum], float)
    support = {"ipcw_evaluation_row_count": len(supported_rows), "ipcw_excluded_row_count": len(rows) - len(supported_rows)}
    if len(times) < 2:
        return {"ipcw_c_index": c_index, "integrated_brier_score": None, "horizon_brier": {}, "calibration": {}, **support}
    probabilities = _curve_values(model, matrix, times)
    _, scores = brier_score(train_y, evaluation_y, probabilities, times)
    return {
        "ipcw_c_index": c_index,
        "integrated_brier_score": float(integrated_brier_score(train_y, evaluation_y, probabilities, times)),
        "horizon_brier": {str(int(time)): float(score) for time, score in zip(times, scores) if int(time) in REPORT_HORIZONS},
        "calibration": {str(int(time)): {"ipcw_brier_score": float(score)} for time, score in zip(times, scores) if int(time) in REPORT_HORIZONS},
        **support,
    }


def select_model(train_rows, validation_rows, feature_columns=RUL_FEATURES, *, native_threads=None):
    train_x, validation_x, train_y = _matrix(train_rows, feature_columns), _matrix(validation_rows, feature_columns), _target(train_rows)
    family_results, fitted = {}, {}
    for family in FAMILY_ORDER:
        candidates = []
        for config in MODEL_FAMILIES[family]:
            model = _model(family, config, native_threads=native_threads).fit(train_x, train_y)
            candidates.append({"config_id": config["id"], "config": config, "validation": _metrics(train_rows, validation_rows, model, feature_columns), "model": model})
        best = min(candidates, key=lambda candidate: (candidate["validation"].get("integrated_brier_score") is None, candidate["validation"].get("integrated_brier_score", float("inf")), -candidate["validation"]["ipcw_c_index"], candidate["config_id"]))
        family_results[family] = {key: value for key, value in best.items() if key != "model"}
        fitted[family] = best["model"]
    winner = select_validation_winner(family_results)
    return fitted[winner["family"]], winner, family_results


def survival_generation_plan(manifest, state_manifest, generation, *, root=ROOT):
    """Create a survival plan from the shared authoritative stream state only."""
    model_config = {"families": MODEL_FAMILIES, "landmark_sampling": LANDMARK_SAMPLING, "ipcw_support_policy": IPCW_SUPPORT_POLICY}
    progressive = state_manifest.get("generation_semantics_version") == PROGRESSIVE_SEMANTICS_VERSION
    plan = (build_progressive_plan(generation, manifest, state_manifest, model_config=model_config,
        feature_version=FEATURE_VERSION, artifact_root=root)
        if progressive else build_generation_plan(generation, manifest, state_manifest,
        model_config=model_config, feature_version=FEATURE_VERSION, artifact_root=root))
    fingerprint = plan["fingerprint"]
    return {**plan, "model_version": f"matr-survival-model-{generation}-{fingerprint[:12]}",
            "artifact_dir": Path(root) / "survival_generations" / fingerprint, "landmark_sampling": LANDMARK_SAMPLING,
            **({"scheduled_manifest_path": Path(root) / state_manifest["scheduled_arrival_manifest_ref"]} if progressive else {})}


def state_bound_lifecycle(lifecycle, plan):
    """Freeze training events to the receipt cohort while retaining held-out facts."""
    arrived = set(plan["arrived_train_battery_ids"])
    retained = [row for row in lifecycle if row["battery_id"] not in arrived]
    return retained + [
        {"battery_id": battery_id, "eol_observed": True}
        for battery_id in plan["observed_eol_train_battery_ids"]
    ]


def plan_from_receipt(receipt_path, manifest, *, root=ROOT):
    """Build the Survival plan only after validating the shared immutable receipt."""
    receipt = read_receipt(receipt_path)
    state = json.loads(Path(receipt["state_manifest_path"]).read_text())
    plan = survival_generation_plan(manifest, state, receipt["generation"], root=root)
    if receipt["schema_version"] == RECEIPT_SCHEMA_VERSION:
        plan["shared_feature_rows"] = load_receipt_feature_rows(receipt, root=root)
        plan["shared_feature_metadata"] = {
            field: receipt[field] for field in (
                "generation_id", "feature_contract_version", "canonical_source_fingerprint",
                "selected_row_count", "selected_rows_sha256",
            )
        }
    else:
        plan["training_features_path"] = Path(receipt["training_features_path"])
        plan["shared_feature_metadata"] = receipt["training_features"]
    return plan


def train_generation(plan, *, root=ROOT, evaluated_at=None):
    """Fit one isolated candidate and score only each battery's latest landmark."""
    if plan is None:
        return None
    root, artifact_dir = Path(root), Path(plan["artifact_dir"])
    evaluation_path = artifact_dir / "candidate_survival_model_evaluation.parquet"
    if evaluation_path.exists():
        return artifact_dir
    features = (list(plan["shared_feature_rows"]) if plan.get("shared_feature_rows") is not None
                else ds.dataset(root / "degradation_features", format="parquet").to_table().to_pylist())
    manifest = pq.read_table(plan.get("scheduled_manifest_path", root / "arrival_manifest.parquet")).to_pylist()
    lifecycle = pq.read_table(root / "replay_lifecycle_state").to_pylist()
    if plan.get("shared_feature_metadata"):
        lifecycle = state_bound_lifecycle(lifecycle, plan)
    if plan.get("training_features_path"):
        split_by_battery = {row["battery_id"]: row["split"] for row in manifest}
        state_features = ds.dataset(plan["training_features_path"], format="parquet").to_table().to_pylist()
        features = [row for row in features if split_by_battery.get(row["battery_id"]) != "train"] + [
            row for row in state_features if row["battery_id"] in set(plan["arrived_train_battery_ids"])
        ]
    full_rows = landmark_rows(features, manifest, lifecycle, plan["cutoff"], train_battery_ids=plan["arrived_train_battery_ids"], fixed_evaluation=True)
    rows = training_landmark_rows(full_rows)
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    full_row_counts = {split: sum(row["split"] == split for row in full_rows) for split in by_split}
    if not all(by_split.values()) or not any(row["event_observed"] for row in by_split["train"]):
        raise ValueError("survival candidate requires train, validation, test, and one observed training event")
    model, winner, family_results = select_model(
        by_split["train"], by_split["validation"], native_threads=plan.get("native_threads"),
    )
    score = {"train": _metrics(by_split["train"], by_split["train"], model, RUL_FEATURES), "validation": winner["validation"], "family_validation": family_results, "selection": {"selected_family": winner["family"], "selected_config_id": winner["config_id"], **SELECTION_POLICY}, "test": _metrics(by_split["train"], by_split["test"], model, RUL_FEATURES)}
    latest = {}
    for row in full_rows:
        if row["cycle_index"] >= latest.get(row["battery_id"], {"cycle_index": -1})["cycle_index"]:
            latest[row["battery_id"]] = row
    evaluated_at = evaluated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    latest_rows = list(latest.values())
    curves = _curve_values(model, _matrix(latest_rows, RUL_FEATURES), HORIZON_GRID)
    predictions = [
        {"model_version": plan["model_version"], "dataset": row["dataset"], "battery_id": row["battery_id"], "cycle_index": row["cycle_index"], "horizon_cycles": horizon, "survival_probability": float(probability), "prediction_created_at": evaluated_at, "split": row["split"]}
        for row, curve in zip(latest_rows, curves) for horizon, probability in zip(HORIZON_GRID, curve)
    ]
    metadata = {"generation": plan["generation"], "model_fingerprint": plan["fingerprint"], "training_battery_count": plan["training_battery_count"], "as_of_cutoff": plan["cutoff"].isoformat(), "event_definition": "observed_eol_only", "censoring_definition": "cutoff_or_replay_end", "feature_version": FEATURE_VERSION, "landmark_sampling": plan["landmark_sampling"], "ipcw_support_policy": IPCW_SUPPORT_POLICY, "selected_family": winner["family"], "selected_config_id": winner["config_id"], "row_counts": {split: len(values) for split, values in by_split.items()}, "full_row_counts": full_row_counts, "event_rows": sum(row["event_observed"] for row in by_split["train"]), "censored_rows": sum(not row["event_observed"] for row in by_split["train"]), "generation_semantics_version": plan["generation_semantics_version"], "snapshot_id": plan["snapshot_id"], "finalized_cycle_boundary_fingerprint": plan["state_manifest"]["finalized_cycle_boundary_fingerprint"], "cohort_checksums": plan["cohort_checksums"], "arrived_train_battery_count": len(plan["arrived_train_battery_ids"]), "observed_eol_train_battery_count": len(plan["observed_eol_train_battery_ids"]), **({"shared_feature_metadata": plan["shared_feature_metadata"]} if plan.get("shared_feature_metadata") else {}), **({"native_threads": plan["native_threads"]} if plan.get("native_threads") is not None else {})}
    if plan["generation_semantics_version"] == PROGRESSIVE_SEMANTICS_VERSION:
        metadata["record_class"] = plan["state_manifest"]["record_class"]
        metadata["progressive_arrival_registry"] = plan["state_manifest"]["progressive_arrival_registry"]
    evaluation = {"model_version": plan["model_version"], "model_name": winner["family"], "dataset": "MATR", "status": "candidate", "evaluated_at": evaluated_at, "model_fingerprint": plan["fingerprint"], "generation": plan["generation"], "metrics_json": json.dumps(score, sort_keys=True), "training_metadata_json": json.dumps(metadata, sort_keys=True)}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, artifact_dir / "selected_model.joblib")
    pq.write_table(pa.Table.from_pylist(predictions), artifact_dir / "candidate_survival_predictions.parquet")
    pq.write_table(pa.Table.from_pylist([evaluation]), evaluation_path)
    return artifact_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a landmark survival candidate.")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "arrival_manifest.parquet")
    parser.add_argument("--lifecycle-events", type=Path, default=ROOT / "replay_lifecycle_state")
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--generation")
    parser.add_argument("--training-features", type=Path)
    parser.add_argument("--receipt", type=Path, help="canonical immutable shared-generation receipt")
    parser.add_argument("--offline-backfill", action="store_true", help="allow the explicit historical reconstruction path")
    parser.add_argument("--native-threads", type=int, help="limit native model workers for a bounded Airflow task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = pq.read_table(args.manifest).to_pylist()
    if args.receipt and (args.state_manifest or args.generation or args.training_features or args.continuous):
        raise SystemExit("--receipt cannot be combined with direct state, feature, or continuous arguments")
    if bool(args.state_manifest) != bool(args.generation):
        raise SystemExit("--state-manifest and --generation must be supplied together")
    if not args.receipt and not (args.offline_backfill and args.state_manifest):
        raise SystemExit("canonical training requires --receipt; use --offline-backfill for historical reconstruction")
    plan = (plan_from_receipt(args.receipt, manifest) if args.receipt else
            survival_generation_plan(manifest, json.loads(args.state_manifest.read_text()), args.generation))
    if args.training_features:
        plan["training_features_path"] = args.training_features
    if args.native_threads is not None:
        plan["native_threads"] = args.native_threads
    artifact = train_generation(plan)
    print(f"Published survival candidate artifact {artifact}" if artifact else "No survival generation due.")
