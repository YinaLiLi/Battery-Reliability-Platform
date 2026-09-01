"""Physical constraints for served RUL predictions."""


def constrain_prediction_row(row):
    constrained = dict(row)
    raw = float(constrained.get("raw_predicted_rul_cycles", constrained["predicted_rul_cycles"]))
    constrained["raw_predicted_rul_cycles"] = raw
    constrained["predicted_rul_cycles"] = max(0.0, raw)
    return constrained


def estimated_eol_cycle(current_cycle, predicted_rul_cycles):
    return current_cycle + max(0.0, float(predicted_rul_cycles))
