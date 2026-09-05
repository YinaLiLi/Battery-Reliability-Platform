"""Survival curve contract shared by training and serving without Spark."""

HORIZON_GRID = tuple(range(0, 201, 10))
REPORT_HORIZONS = (50, 100, 200)


def validate_prediction_rows(rows):
    ordered = sorted(rows, key=lambda row: row["horizon_cycles"])
    if not set(REPORT_HORIZONS).issubset({row["horizon_cycles"] for row in ordered}):
        raise ValueError("survival predictions are missing required horizons")
    previous = 1.0
    for row in ordered:
        probability = float(row["survival_probability"])
        if not 0 <= probability <= 1:
            raise ValueError("survival probability must be between zero and one")
        if row["horizon_cycles"] == 0 and probability != 1.0:
            raise ValueError("survival probability at zero must equal one")
        if probability > previous:
            raise ValueError("survival probabilities must be non-increasing")
        previous = probability
