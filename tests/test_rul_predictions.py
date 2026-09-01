from src.rul_predictions import constrain_prediction_row, estimated_eol_cycle


def test_constrain_prediction_row_preserves_raw_output_and_clamps_served_rul():
    row = constrain_prediction_row({"predicted_rul_cycles": -4.5})

    assert row == {"raw_predicted_rul_cycles": -4.5, "predicted_rul_cycles": 0.0}


def test_estimated_eol_never_precedes_current_cycle():
    assert estimated_eol_cycle(120, -4.5) == 120.0
    assert estimated_eol_cycle(120, 25.0) == 145.0
