"""Physical constraints for served RUL predictions."""


def constrain_prediction_row(row):
    constrained = dict(row)
    raw = float(constrained.get("raw_predicted_rul_cycles", constrained["predicted_rul_cycles"]))
    constrained["raw_predicted_rul_cycles"] = raw
    constrained["predicted_rul_cycles"] = max(0.0, raw)
    return constrained


def constrain_prediction_trajectory(rows):
    """Freeze a battery's served EOL at its first zero-RUL prediction."""
    eol_cycle = None
    constrained = []
    for row in sorted(rows, key=lambda item: item["cycle_index"]):
        served = constrain_prediction_row(row)
        cycle = float(served["cycle_index"])
        if eol_cycle is None and served["predicted_rul_cycles"] == 0:
            eol_cycle = cycle
        if eol_cycle is not None:
            served["predicted_rul_cycles"] = 0.0
            served["predicted_eol_cycle"] = eol_cycle
        else:
            served["predicted_eol_cycle"] = cycle + served["predicted_rul_cycles"]
        constrained.append(served)
    return constrained


def estimated_eol_cycle(current_cycle, predicted_rul_cycles):
    return current_cycle + max(0.0, float(predicted_rul_cycles))
