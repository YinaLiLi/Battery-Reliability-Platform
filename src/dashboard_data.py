"""Small presentation helpers for the read-only reliability dashboard."""
import json


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
    stage = metrics.get("lifecycle_stage_mae", {})
    metadata = evaluation.get("training_metadata") or {}
    training_data = metadata.get("training_data_version", "Not recorded")
    return {
        "Model version": evaluation["model_version"],
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
