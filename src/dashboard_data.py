"""Small presentation helpers for the read-only reliability dashboard."""
import json
import math

import pandas as pd


FAMILY_LABELS = {"cox": "Cox", "random_survival_forest": "Random Survival Forest", "ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}
LEGACY_XGB_NAMES = {
    "xgboost ruler regressor",
    "xgboost rul regressor",
    "xgboostrulregressor",
    "xgboost_rul_regressor",
    "xgboost-rul-regressor",
}


def performance_gradient(frame, *, lower_is_better=(), higher_is_better=()):
    """Return a Styler where darker cells always mean better performance."""
    style = frame.style
    palette = ("#eff6ff", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a")
    for column, lower in [(column, True) for column in lower_is_better] + [(column, False) for column in higher_is_better]:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        if finite.max() == finite.min():
            score = values.where(values.isna(), 1.0)
        elif lower:
            score = (finite.max() - values) / (finite.max() - finite.min())
        else:
            score = (values - finite.min()) / (finite.max() - finite.min())
        def css(values, score=score):
            return [
                "" if pd.isna(value) else f"background-color: {palette[min(len(palette) - 1, int(round(float(score.iloc[index]) * (len(palette) - 1))))]}; color: {'white' if score.iloc[index] >= 0.6 else '#0f172a'}"
                for index, value in enumerate(values)
            ]
        style = style.apply(css, subset=[column])
    return style


def family_label(family):
    return FAMILY_LABELS.get(family, str(family).replace("_", " ").title())


def _canonical_family_hint(value):
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized


def soh_percent(soh):
    return round(soh * 100, 2)


def measured_soh_distribution(fleet, bin_width=5):
    """Count observed current-battery SOH measurements in fixed percentage bins."""
    values = pd.to_numeric(fleet["measured_soh"], errors="coerce").dropna() * 100
    if values.empty:
        return pd.DataFrame(columns=["Measured SOH bin", "Battery count"])
    lower = math.floor(values.min() / bin_width) * bin_width
    upper = math.ceil(values.max() / bin_width) * bin_width
    if upper <= values.max():
        upper += bin_width
    bins = list(range(int(lower), int(upper) + bin_width, bin_width))
    counts = pd.cut(values, bins=bins, right=False, include_lowest=True).value_counts(sort=False)
    counts = counts[counts > 0]
    return pd.DataFrame(
        {
            "Measured SOH bin": [f"{interval.left:.0f}–{interval.right:.0f}%" for interval in counts.index],
            "Battery count": counts.to_numpy(),
        }
    )


def lowest_rows(rows, field, limit=5):
    return sorted((row for row in rows if row.get(field) is not None), key=lambda row: row[field])[:limit]


def lifecycle_stage(current_cycle, predicted_rul_cycles):
    """Classify progress using the model-estimated EOL only."""
    if predicted_rul_cycles is None:
        return "Unavailable"
    estimated_eol = current_cycle + predicted_rul_cycles
    progress = current_cycle / estimated_eol if estimated_eol else 0
    if progress < 1 / 3:
        return "Early"
    if progress < 2 / 3:
        return "Mid"
    return "Late"


def _training_metadata(model):
    metadata = model.get("training_metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata


def _model_generation(model):
    return model.get("generation") or _training_metadata(model).get("generation")


def _normalized_model_family(model):
    metadata = _training_metadata(model)
    family = metadata.get("selected_family")
    if not family:
        family = model.get("model_name")
    normalized = _canonical_family_hint(family) if family else "not_recorded"
    if normalized in LEGACY_XGB_NAMES:
        return "xgboost"
    return normalized


def _is_legacy_xgb_record(model):
    model_version = str(model.get("model_version") or "")
    family = _canonical_family_hint(model.get("model_name"))
    return model_version.startswith("matr-rul-xgboost-") and family in LEGACY_XGB_NAMES


def model_metrics(evaluation):
    """Flatten a serving evaluation row for table display."""
    metrics = evaluation["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    test = metrics.get("test", {})
    validation = metrics.get("validation", {})
    stage = metrics.get("lifecycle_stage_mae", {})
    metadata = evaluation.get("training_metadata") or {}
    training_data = metadata.get("training_data_version", "Not recorded")
    return {
        "Model version": evaluation["model_version"],
        "Generation": evaluation.get("generation") or metadata.get("generation", "Not recorded"),
        "Selected model family": family_label(_normalized_model_family(evaluation)),
        "Training batteries": metadata.get("training_battery_count", "Not recorded"),
        "Model fingerprint": evaluation.get("model_fingerprint") or metadata.get("model_fingerprint", "Not recorded"),
        "Validation MAE": validation.get("mae"),
        "Status": evaluation["status"],
        "Evaluated at": str(evaluation["evaluated_at"]),
        "Test MAE": test.get("mae"),
        "Test RMSE": test.get("rmse"),
        "Test R²": test.get("r2"),
        "Early MAE": stage.get("early"),
        "Mid MAE": stage.get("mid"),
        "Late MAE": stage.get("late"),
        "Training data": training_data,
    }


def survival_model_metrics(evaluation):
    """Flatten survival evaluation without introducing RUL error metrics."""
    metrics = evaluation["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    validation, test = metrics.get("validation", {}), metrics.get("test", {})
    metadata = evaluation.get("training_metadata") or {}
    return {
        "Model version": evaluation["model_version"], "Generation": evaluation.get("generation") or metadata.get("generation", "Not recorded"),
        "Selected model family": family_label(metadata.get("selected_family") or evaluation.get("model_name")), "Training batteries": metadata.get("training_battery_count", "Not recorded"),
        "Status": evaluation["status"], "Evaluated at": str(evaluation["evaluated_at"]),
        "Validation IBS": validation.get("integrated_brier_score"), "Validation IPCW C-index": validation.get("ipcw_c_index"),
        "Test IBS": test.get("integrated_brier_score"), "Test IPCW C-index": test.get("ipcw_c_index"),
    }


def survival_family_validation_rows(evaluation):
    metrics = evaluation.get("metrics") or {}
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    selected = metrics.get("selection", {}).get("selected_family") or (evaluation.get("training_metadata") or {}).get("selected_family") or evaluation.get("model_name")
    return [
        {
            "Model family": family_label(family),
            "Validation IBS": result.get("validation", {}).get("integrated_brier_score"),
            "Validation IPCW C-index": result.get("validation", {}).get("ipcw_c_index"),
            "Configuration": result.get("config_id"),
            "Selected": family == selected,
        }
        for family, result in metrics.get("family_validation", {}).items()
    ]


def model_display_names(models):
    """Return stable short labels while retaining internal IDs as metadata."""
    active = [model for model in models if model["status"] in {"candidate", "champion"}]
    if not any((model.get("generation") or (model.get("training_metadata") or {}).get("generation")) is not None for model in active):
        champion = next((model for model in active if model["status"] == "champion"), None)
        names = {champion["model_version"]: "XGBoost 1.0"} if champion else {}
        candidates = sorted((model for model in active if model["status"] == "candidate"), key=lambda model: (str(model.get("evaluated_at", "")), model["model_version"]))
        names.update({model["model_version"]: f"XGBoost 1.{index + 1}" for index, model in enumerate(candidates)})
        return names
    return {
        model["model_version"]: f"Model {model.get('generation') or (model.get('training_metadata') or {}).get('generation')} — {family_label(_normalized_model_family(model))}"
        for model in active
    }


def family_validation_rows(evaluation):
    metrics = evaluation["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    selected = _canonical_family_hint(
        metrics.get("selection", {}).get("selected_family") or (evaluation.get("training_metadata") or {}).get("selected_family") or evaluation.get("model_name", "")
    )
    if selected in LEGACY_XGB_NAMES:
        selected = "xgboost"
    return [
        {
            "Model family": family_label("xgboost" if _canonical_family_hint(_canonical_model_family) in LEGACY_XGB_NAMES else _canonical_family_hint(_canonical_model_family)),
            "Selected": _canonical_family_hint(_canonical_model_family) == selected,
            "Configuration": result.get("config_id"),
            "Validation MAE": result.get("validation", {}).get("mae"),
            "Validation RMSE": result.get("validation", {}).get("rmse"),
            "Validation R²": result.get("validation", {}).get("r2"),
        }
        for _canonical_model_family, result in metrics.get("family_validation", {}).items()
    ]


def selectable_models(models):
    """Return non-retired generations in display order."""
    active = [model for model in models if model.get("status") != "retired"]
    generations = {}
    for model in active:
        if _is_legacy_xgb_record(model):
            continue
        generation = _model_generation(model)
        if generation is None:
            continue
        current = generations.get(generation)
        if current is None:
            generations[generation] = model
            continue
        current_metadata = _training_metadata(current)
        next_metadata = _training_metadata(model)
        def canonical_rank(metadata):
            if metadata.get("record_class") == "canonical_shared_progressive_arrival_v3":
                return 3
            if metadata.get("generation_semantics_version") == "shared-stream-state-v2":
                return 1
            return 2
        current_rank, next_rank = canonical_rank(current_metadata), canonical_rank(next_metadata)
        if next_rank != current_rank:
            if next_rank > current_rank:
                generations[generation] = model
            continue
        current_is_neutral = bool(current_metadata.get("selected_family"))
        next_is_neutral = bool(next_metadata.get("selected_family"))
        if next_is_neutral and not current_is_neutral:
            generations[generation] = model
            continue
        if next_is_neutral == current_is_neutral and model.get("status") == "champion" and current.get("status") != "champion":
            generations[generation] = model
            continue
        if next_is_neutral == current_is_neutral and model.get("evaluated_at") > current.get("evaluated_at", ""):
            generations[generation] = model
    selected = list(generations.values())
    names = model_display_names(active)
    return sorted(selected, key=lambda model: names.get(model["model_version"], model["model_version"]))


def latest_model_version(models):
    """Return the best fixed-test-MAE selectable generation, if recorded."""
    active = selectable_models(models)
    if not active:
        return None
    preferred = next((model for model in active if str(_model_generation(model)) == "1.2"), None)
    if preferred is not None:
        return preferred["model_version"]

    def test_mae(model):
        metrics = model.get("metrics") or {}
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        return metrics.get("test", {}).get("mae", float("inf"))
    return min(active, key=lambda model: (test_mae(model), model["model_version"]))["model_version"]
