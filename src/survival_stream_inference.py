"""Current conditional-survival rows from finalized causal features."""
from datetime import datetime, timezone

import numpy as np

try:
    from .survival_contract import HORIZON_GRID, validate_prediction_rows
except ImportError:
    from survival_contract import HORIZON_GRID, validate_prediction_rows


def current_survival_rows(model, rows, *, feature_columns, model_version, model_fingerprint,
                          state_id, feature_contract_version, selection_revision, created_at=None):
    """Score already-filtered latest feature rows on the fixed serving grid."""
    if not rows:
        return []
    matrix = np.asarray([[np.nan if row.get(column) is None else row[column] for column in feature_columns] for row in rows], float)
    matrix[~np.isfinite(matrix)] = np.nan
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    output = []
    for row, curve in zip(rows, model.predict_survival_function(matrix)):
        previous, battery_rows = 1.0, []
        for horizon in HORIZON_GRID:
            probability = 1.0 if horizon == 0 else float(curve(horizon))
            probability = min(previous, max(0.0, min(1.0, probability))) if np.isfinite(probability) else 0.0
            battery_rows.append({
                "dataset": row["dataset"], "battery_id": row["battery_id"], "cycle_index": int(row["cycle_index"]),
                "horizon_cycles": horizon, "survival_probability": probability, "model_version": model_version,
                "model_fingerprint": model_fingerprint, "state_id": state_id,
                "replay_sequence": int(row.get("replay_sequence", 0)),
                "feature_contract_version": feature_contract_version, "selection_revision": int(selection_revision),
                "inference_created_at": created_at,
            })
            previous = probability
        validate_prediction_rows(battery_rows)
        output.extend(battery_rows)
    return output
