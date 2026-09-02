"""Small presentation helpers for the read-only reliability dashboard."""
import json


FAMILY_LABELS = {"ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}


def family_label(family):
    return FAMILY_LABELS.get(family, str(family).replace("_", " ").title())


def soh_percent(soh):
    return round(soh * 100, 2)


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
        "Selected model family": family_label(metadata.get("selected_family") or evaluation.get("model_name", "Not recorded")),
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
        model["model_version"]: f"Model {model.get('generation') or (model.get('training_metadata') or {}).get('generation')} — {family_label((model.get('training_metadata') or {}).get('selected_family') or model.get('model_name'))}"
        for model in active
    }


def family_validation_rows(evaluation):
    metrics = evaluation["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    selected = metrics.get("selection", {}).get("selected_family") or (evaluation.get("training_metadata") or {}).get("selected_family") or evaluation.get("model_name")
    return [
        {
            "Model family": family_label(family),
            "Selected": family == selected,
            "Configuration": result.get("config_id"),
            "Validation MAE": result.get("validation", {}).get("mae"),
            "Validation RMSE": result.get("validation", {}).get("rmse"),
            "Validation R²": result.get("validation", {}).get("r2"),
        }
        for family, result in metrics.get("family_validation", {}).items()
    ]


def selectable_models(models):
    """Return non-retired generations in display order."""
    active = [model for model in models if model.get("status") != "retired"]
    names = model_display_names(active)
    return sorted(active, key=lambda model: names.get(model["model_version"], model["model_version"]))


def latest_model_version(models):
    """Return the newest selectable generation, if any."""
    active = selectable_models(models)
    return active[-1]["model_version"] if active else None
