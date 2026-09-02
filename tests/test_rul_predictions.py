from src.rul_predictions import constrain_prediction_row, constrain_prediction_trajectory, estimated_eol_cycle


def test_constrain_prediction_row_preserves_raw_output_and_clamps_served_rul():
    row = constrain_prediction_row({"predicted_rul_cycles": -4.5})

    assert row == {"raw_predicted_rul_cycles": -4.5, "predicted_rul_cycles": 0.0}


def test_estimated_eol_never_precedes_current_cycle():
    assert estimated_eol_cycle(120, -4.5) == 120.0
    assert estimated_eol_cycle(120, 25.0) == 145.0


def test_prediction_trajectory_freezes_served_eol_after_first_zero():
    rows = constrain_prediction_trajectory([
        {"cycle_index": 600, "predicted_rul_cycles": 10.0},
        {"cycle_index": 540, "predicted_rul_cycles": 20.0},
        {"cycle_index": 560, "predicted_rul_cycles": -2.0},
        {"cycle_index": 700, "predicted_rul_cycles": 30.0},
    ])

    assert [(row["cycle_index"], row["predicted_rul_cycles"], row["predicted_eol_cycle"]) for row in rows] == [
        (540, 20.0, 560.0), (560, 0.0, 560.0), (600, 0.0, 560.0), (700, 0.0, 560.0)
    ]
    assert rows[2]["raw_predicted_rul_cycles"] == 10.0
