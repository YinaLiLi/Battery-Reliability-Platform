"""Pure, model-pinned scoring for finalized streaming feature rows."""

from datetime import datetime, timezone

try:
    from .rul_predictions import constrain_prediction_row
except ImportError:
    from rul_predictions import constrain_prediction_row


def current_prediction_rows(model, rows, *, feature_columns, model_version, model_fingerprint, state_id, selection_revision, benchmark_battery_ids=(), prior_predictions=None, created_at=None):
    """Score non-benchmark finalized rows and retain an already-frozen EOL."""
    prior_predictions = prior_predictions or {}
    created_at = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    eligible = [row for row in rows if row["battery_id"] not in set(benchmark_battery_ids)]
    matrix = [[row.get(column) for column in feature_columns] for row in eligible]
    predictions = model.predict(matrix) if matrix else []
    output = []
    for row, raw in zip(eligible, predictions):
        served = constrain_prediction_row({"predicted_rul_cycles": float(raw)})
        previous = prior_predictions.get(row["battery_id"])
        cycle = float(row["cycle_index"])
        if previous and float(previous.get("predicted_rul_cycles", 1)) == 0:
            served["predicted_rul_cycles"] = 0.0
            served["predicted_eol_cycle"] = float(previous["predicted_eol_cycle"])
        else:
            served["predicted_eol_cycle"] = cycle + served["predicted_rul_cycles"]
        output.append({
            "dataset": row["dataset"], "battery_id": row["battery_id"], "model_version": model_version,
            "model_fingerprint": model_fingerprint, "state_id": state_id,
            "replay_sequence": int(row["replay_sequence"]), "cycle_index": int(row["cycle_index"]),
            "raw_predicted_rul_cycles": served["raw_predicted_rul_cycles"],
            "predicted_rul_cycles": served["predicted_rul_cycles"], "predicted_eol_cycle": served["predicted_eol_cycle"],
            "inference_created_at": created_at, "selection_revision": int(selection_revision),
        })
    return output
