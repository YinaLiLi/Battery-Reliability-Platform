"""Compare local battery-failure classifiers without changing the Spark feature job."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURE_PATH = Path("data/processed/spark_batch_features")
DEFAULT_REPORT_PATH = Path("data/processed/primary_model_comparison_metrics.json")
DEFAULT_TEMPORAL_REPORT_PATH = Path("data/processed/temporal_stress_metrics.json")
TARGET = "failure_within_30_operating_days"
DEFAULT_THRESHOLD = 0.50
IDENTIFIER_COLUMNS = ("event_id", "vehicle_id", "timestamp", TARGET, "vehicle_battery_type", "vehicle_region")
NUMERIC_COLUMNS = (
    "battery_age_days",
    "soc",
    "pack_voltage",
    "pack_current",
    "module_temp_min",
    "module_temp_max",
    "outside_temp",
    "odometer",
    "module_temp_spread",
    "previous_pack_voltage",
    "rolling_module_temp_max",
)
CATEGORICAL_COLUMNS = ("battery_type", "region", "is_charging")
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
PRIMARY_START = datetime(2025, 1, 1)
PRIMARY_END = datetime(2025, 3, 31, 23, 59, 59)
TEMPORAL_BOUNDARIES = {
    "train_end": datetime(2025, 3, 1, 23, 59, 59),
    "validation_end": datetime(2025, 3, 15, 23, 59, 59),
    "test_end": datetime(2025, 3, 31, 23, 59, 59),
}
MIN_CLASS_VEHICLES = 2
MIN_CLASS_ROWS = 2
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}
TIMING_BANDS = ("no_eol", "early_positive_window", "mid_positive_window", "late_positive_window", "after_primary_horizon")
EARLY_POSITIVE_END = datetime(2025, 2, 14, 23, 59, 59)
MID_POSITIVE_END = datetime(2025, 3, 14, 23, 59, 59)


@dataclass
class FeatureTransformer:
    scale_numeric: bool

    def fit(self, rows):
        numeric = _numeric_matrix(rows)
        categorical = _categorical_matrix(rows)
        self.imputer = SimpleImputer(strategy="median").fit(numeric)
        imputed = self.imputer.transform(numeric)
        self.scaler = StandardScaler().fit(imputed) if self.scale_numeric else None
        self.encoder = OneHotEncoder(handle_unknown="ignore").fit(categorical)
        return self

    def transform(self, rows):
        numeric = self.imputer.transform(_numeric_matrix(rows))
        if self.scaler is not None:
            numeric = self.scaler.transform(numeric)
        return sparse.hstack((sparse.csr_matrix(numeric), self.encoder.transform(_categorical_matrix(rows))), format="csr")


def _numeric_matrix(rows):
    return np.asarray([[row.get(column, np.nan) for column in NUMERIC_COLUMNS] for row in rows], dtype=float)


def _categorical_matrix(rows):
    return np.asarray([[str(row.get(column, "missing")) for column in CATEGORICAL_COLUMNS] for row in rows], dtype=object)


def load_rows(feature_path):
    columns = ("vehicle_id", "timestamp", TARGET, "battery_type", "region") + FEATURE_COLUMNS
    return ds.dataset(str(feature_path), format="parquet", partitioning="hive").to_table(columns=list(dict.fromkeys(columns))).to_pylist()


def vehicle_timing_band(vehicle_rows):
    """Classify a vehicle by the first observed 30-day positive-risk label."""
    positives = [row["timestamp"] for row in vehicle_rows if row[TARGET]]
    if not positives:
        return "no_eol"
    first_positive = min(positives)
    if first_positive <= EARLY_POSITIVE_END:
        return "early_positive_window"
    if first_positive <= MID_POSITIVE_END:
        return "mid_positive_window"
    if first_positive <= PRIMARY_END:
        return "late_positive_window"
    return "after_primary_horizon"


def _vehicle_split_table(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["vehicle_id"]].append(row)
    return {
        vehicle_id: {
            "vehicle_id": vehicle_id,
            "region": vehicle_rows[0]["region"],
            "battery_type": vehicle_rows[0]["battery_type"],
            "timing_band": vehicle_timing_band(vehicle_rows),
        }
        for vehicle_id, vehicle_rows in grouped.items()
    }


def _cohort_sizes(total):
    raw_sizes = {name: total * SPLIT_RATIOS[name] for name in SPLIT_NAMES}
    sizes = {name: int(raw_sizes[name]) for name in SPLIT_NAMES}
    for name in sorted(SPLIT_NAMES, key=lambda item: (-(raw_sizes[item] - sizes[item]), SPLIT_NAMES.index(item)))[: total - sum(sizes.values())]:
        sizes[name] += 1
    return sizes


def _vehicle_cohorts(rows, seed=42):
    """Allocate exact cohort sizes while keeping categorical timing quotas close."""
    vehicles = _vehicle_split_table(rows)
    attributes = ("region", "battery_type", "timing_band", "stratum")
    weights = {"region": 2, "battery_type": 2, "timing_band": 4, "stratum": 1}
    for vehicle in vehicles.values():
        vehicle["stratum"] = (vehicle["region"], vehicle["battery_type"], vehicle["timing_band"])
    totals = {attribute: Counter(vehicle[attribute] for vehicle in vehicles.values()) for attribute in attributes}
    counts = {name: {attribute: Counter() for attribute in attributes} for name in SPLIT_NAMES}
    sizes = _cohort_sizes(len(vehicles))
    assigned = {name: set() for name in SPLIT_NAMES}

    def stable_order(item):
        vehicle_id, vehicle = item
        return (totals["stratum"][vehicle["stratum"]], hashlib.sha256(f"{seed}:{vehicle_id}".encode()).hexdigest(), vehicle_id)

    for vehicle_id, vehicle in sorted(vehicles.items(), key=stable_order):
        candidates = [name for name in SPLIT_NAMES if len(assigned[name]) < sizes[name]]

        def quota_error(name):
            error = 0.0
            for attribute in attributes:
                value = vehicle[attribute]
                target = totals[attribute][value] * SPLIT_RATIOS[name]
                current = counts[name][attribute][value]
                error += weights[attribute] * (((current + 1 - target) ** 2 - (current - target) ** 2) / max(target, 1))
            return error

        split = min(candidates, key=lambda name: (quota_error(name), SPLIT_NAMES.index(name)))
        assigned[split].add(vehicle_id)
        for attribute in attributes:
            counts[split][attribute][vehicle[attribute]] += 1
    return assigned


def _validate_class_coverage(splits, min_class_vehicles, min_class_rows):
    for name, rows in splits.items():
        labels = {row[TARGET] for row in rows}
        if labels != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
        for label in (0, 1):
            class_rows = [row for row in rows if row[TARGET] == label]
            if len(class_rows) < min_class_rows or len({row["vehicle_id"] for row in class_rows}) < min_class_vehicles:
                raise ValueError(f"{name} lacks enough rows or vehicles for class {label}")


def make_primary_splits(rows, seed=42, min_class_vehicles=MIN_CLASS_VEHICLES, min_class_rows=MIN_CLASS_ROWS):
    cohorts = _vehicle_cohorts(rows, seed)
    splits = {
        name: [
            row
            for row in rows
            if row["vehicle_id"] in vehicle_ids and PRIMARY_START <= row["timestamp"] <= PRIMARY_END
        ]
        for name, vehicle_ids in cohorts.items()
    }
    vehicle_sets = {name: {row["vehicle_id"] for row in rows} for name, rows in splits.items()}
    if vehicle_sets["train"] & vehicle_sets["validation"] or vehicle_sets["train"] & vehicle_sets["test"] or vehicle_sets["validation"] & vehicle_sets["test"]:
        raise ValueError("vehicle cohorts overlap")
    _validate_class_coverage(splits, min_class_vehicles, min_class_rows)
    return splits


def make_temporal_splits(rows, train_end=None, validation_end=None, test_end=None, min_class_vehicles=MIN_CLASS_VEHICLES, min_class_rows=MIN_CLASS_ROWS):
    boundaries = {
        "train_end": train_end or TEMPORAL_BOUNDARIES["train_end"],
        "validation_end": validation_end or TEMPORAL_BOUNDARIES["validation_end"],
        "test_end": test_end or TEMPORAL_BOUNDARIES["test_end"],
    }
    if not boundaries["train_end"] < boundaries["validation_end"] < boundaries["test_end"]:
        raise ValueError("temporal boundaries must be chronological")
    splits = {
        "train": [row for row in rows if row["timestamp"] <= boundaries["train_end"]],
        "validation": [row for row in rows if boundaries["train_end"] < row["timestamp"] <= boundaries["validation_end"]],
        "test": [row for row in rows if boundaries["validation_end"] < row["timestamp"] <= boundaries["test_end"]],
    }
    _validate_class_coverage(splits, min_class_vehicles, min_class_rows)
    return splits


def choose_recall_threshold(labels, probabilities, recall_target=0.8):
    from sklearn.metrics import precision_recall_curve

    _, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= recall_target)
    if not len(eligible):
        return float(thresholds[0])
    return float(thresholds[eligible[-1]])


def _metrics(labels, probabilities, threshold):
    predictions = np.asarray(probabilities) >= threshold
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "alert_rate": float(predictions.mean()),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


CANDIDATE_NAMES = (
    "logistic_regression",
    "random_forest_depth_10",
    "random_forest_depth_16",
    "xgboost_depth_3",
    "xgboost_depth_5",
)


def _candidate(name, positive_weight):
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42), True
    if name == "random_forest_depth_10":
        return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42), False
    if name == "random_forest_depth_16":
        return RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42), False
    from xgboost import XGBClassifier

    if name == "xgboost_depth_3":
        return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=positive_weight, early_stopping_rounds=30, eval_metric="logloss", n_jobs=-1, random_state=42), False
    if name == "xgboost_depth_5":
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=positive_weight, early_stopping_rounds=30, eval_metric="logloss", n_jobs=-1, random_state=42), False
    raise ValueError(f"unknown candidate: {name}")


def select_model(results):
    family_order = {"logistic_regression": 0, "random_forest": 1, "xgboost": 2}
    best_pr_auc = max(result["validation"]["pr_auc"] for result in results)
    finalists = [result for result in results if best_pr_auc - result["validation"]["pr_auc"] <= 0.005]
    return min(finalists, key=lambda result: (family_order[next(family for family in family_order if result["name"].startswith(family))], result["name"]))


def split_summary(splits):
    summary = {}
    for name, rows in splits.items():
        positive = sum(row[TARGET] for row in rows)
        summary[name] = {
            "vehicles": len({row["vehicle_id"] for row in rows}),
            "rows": len(rows),
            "positive": positive,
            "negative": len(rows) - positive,
            "positive_rate": positive / len(rows),
        }
    return summary


def _mean_median(values):
    values = [value for value in values if value is not None and not np.isnan(value)]
    return {"mean": float(np.mean(values)), "median": float(np.median(values))} if values else {"mean": None, "median": None}


def _primary_cohort_balance(rows, cohorts):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["vehicle_id"]].append(row)
    vehicles = _vehicle_split_table(rows)
    primary_rows = {
        vehicle_id: [row for row in vehicle_rows if PRIMARY_START <= row["timestamp"] <= PRIMARY_END]
        for vehicle_id, vehicle_rows in grouped.items()
    }
    report = {}
    for name, vehicle_ids in cohorts.items():
        vehicle_ids = sorted(vehicle_ids)
        cohort_vehicles = [vehicles[vehicle_id] for vehicle_id in vehicle_ids]
        vehicle_rows = [primary_rows[vehicle_id] for vehicle_id in vehicle_ids]
        first_positive_dates = []
        vehicle_positive_rates = []
        secondary = defaultdict(list)
        for vehicle_id, rows_for_vehicle in zip(vehicle_ids, vehicle_rows):
            all_rows = grouped[vehicle_id]
            positives = [row["timestamp"] for row in all_rows if row[TARGET]]
            first_positive_dates.append(min(positives).date().isoformat() if positives else "none")
            vehicle_positive_rates.append(sum(row[TARGET] for row in rows_for_vehicle) / len(rows_for_vehicle))
            initial = min(all_rows, key=lambda row: row["timestamp"])
            secondary["initial_battery_age_days"].append(initial.get("battery_age_days"))
            secondary["initial_odometer"].append(initial.get("odometer"))
            for label, column in (("average_soc", "soc"), ("average_pack_voltage", "pack_voltage"), ("average_outside_temp", "outside_temp")):
                secondary[label].append(np.mean([row[column] for row in rows_for_vehicle if row[column] is not None]))
            secondary["average_module_temp"].append(
                np.mean([(row["module_temp_min"] + row["module_temp_max"]) / 2 for row in rows_for_vehicle if row["module_temp_min"] is not None and row["module_temp_max"] is not None])
            )
        report[name] = {
            "region": dict(sorted(Counter(vehicle["region"] for vehicle in cohort_vehicles).items())),
            "battery_type": dict(sorted(Counter(vehicle["battery_type"] for vehicle in cohort_vehicles).items())),
            "timing_band": dict(sorted(Counter(vehicle["timing_band"] for vehicle in cohort_vehicles).items())),
            "ever_eol": {"yes": sum(date != "none" for date in first_positive_dates), "no": sum(date == "none" for date in first_positive_dates)},
            "row_positive_rate": sum(row[TARGET] for rows_for_vehicle in vehicle_rows for row in rows_for_vehicle) / sum(len(rows_for_vehicle) for rows_for_vehicle in vehicle_rows),
            "vehicle_positive_rate": _mean_median(vehicle_positive_rates),
            "first_positive_date": dict(sorted(Counter(first_positive_dates).items())),
            "secondary_balance_checks": {label: _mean_median(values) for label, values in secondary.items()},
        }
    return report


def _feature_shift(splits):
    return {
        "numeric_median": {
            column: {
                name: float(np.nanmedian([np.nan if row[column] is None else row[column] for row in rows]))
                for name, rows in splits.items()
            }
            for column in NUMERIC_COLUMNS
        },
        "categorical_share": {
            column: {
                name: {
                    value: sum(str(row[column]) == value for row in rows) / len(rows)
                    for value in sorted({str(row[column]) for row in rows})
                }
                for name, rows in splits.items()
            }
            for column in CATEGORICAL_COLUMNS
        },
    }


def _fit_candidate(name, train, validation):
    train_labels = np.asarray([row[TARGET] for row in train])
    validation_labels = np.asarray([row[TARGET] for row in validation])
    positive_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    model, scale_numeric = _candidate(name, positive_weight)
    transformer = FeatureTransformer(scale_numeric).fit(train)
    train_features, validation_features = transformer.transform(train), transformer.transform(validation)
    fit_args = {"eval_set": [(validation_features, validation_labels)], "verbose": False} if name.startswith("xgboost") else {}
    model.fit(train_features, train_labels, **fit_args)
    train_probabilities = model.predict_proba(train_features)[:, 1]
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    threshold = DEFAULT_THRESHOLD
    recall_first_threshold = choose_recall_threshold(validation_labels, validation_probabilities)
    train_metrics = _metrics(train_labels, train_probabilities, threshold)
    validation_metrics = _metrics(validation_labels, validation_probabilities, threshold)
    return {
        "name": name,
        "model": model,
        "transformer": transformer,
        "threshold": threshold,
        "train": train_metrics,
        "validation": validation_metrics,
        "recall_first_diagnostic": {
            "threshold": recall_first_threshold,
            "validation": _metrics(validation_labels, validation_probabilities, recall_first_threshold),
        },
        "generalization_gap": {metric: train_metrics[metric] - validation_metrics[metric] for metric in ("roc_auc", "pr_auc", "precision", "recall")},
    }


def _score_test(result, test):
    labels = np.asarray([row[TARGET] for row in test])
    probabilities = result["model"].predict_proba(result["transformer"].transform(test))[:, 1]
    result["test"] = _metrics(labels, probabilities, result["threshold"])


def _public_result(result):
    return {key: value for key, value in result.items() if key not in {"model", "transformer"}}


def run_primary_comparison(rows):
    splits = make_primary_splits(rows)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    results = [_fit_candidate(name, train, validation) for name in CANDIDATE_NAMES]
    selected = select_model(results)
    _score_test(selected, test)
    return {
        "split_strategy": "vehicle_disjoint_common_horizon",
        "horizon": {"start": PRIMARY_START.isoformat(), "end": PRIMARY_END.isoformat()},
        "stratification": "region_x_battery_type_x_first_positive_timing_band",
        "split_seed": 42,
        "threshold_policy": "fixed_probability_0.50",
        "splits": split_summary(splits),
        "cohort_balance": _primary_cohort_balance(rows, _vehicle_cohorts(rows)),
        "selected_model": selected["name"],
        "threshold": selected["threshold"],
        "models": [_public_result(result) for result in results],
    }


def run_temporal_stress(rows, selected_name):
    splits = make_temporal_splits(rows)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    result = _fit_candidate(selected_name, train, validation)
    _score_test(result, test)
    return {
        "split_strategy": "temporal_overlap_stress_test",
        "boundaries": {key: value.isoformat() for key, value in TEMPORAL_BOUNDARIES.items()},
        "threshold_policy": "fixed_probability_0.50",
        "splits": split_summary(splits),
        "selected_model": selected_name,
        "threshold": result["threshold"],
        "model": _public_result(result),
        "feature_distribution_shift": _feature_shift(splits),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare battery-failure classifiers and run a separate temporal stress test.")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--temporal-stress-report-path", type=Path, default=DEFAULT_TEMPORAL_REPORT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.feature_path)
    primary_report = run_primary_comparison(rows)
    temporal_report = run_temporal_stress(rows, primary_report["selected_model"])
    for path, report in ((args.report_path, primary_report), (args.temporal_stress_report_path, temporal_report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"primary": primary_report, "temporal_stress": temporal_report}, indent=2))


if __name__ == "__main__":
    main()
