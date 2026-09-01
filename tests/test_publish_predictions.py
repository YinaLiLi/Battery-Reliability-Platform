import pyarrow as pa

from src.publish_predictions import constrain_prediction_table


def test_publish_constraint_preserves_raw_prediction_and_removes_negative_served_rul():
    source = pa.Table.from_pylist([{"predicted_rul_cycles": -2.0}, {"predicted_rul_cycles": 3.0}])

    constrained = constrain_prediction_table(source).to_pylist()

    assert constrained == [
        {"predicted_rul_cycles": 0.0, "raw_predicted_rul_cycles": -2.0},
        {"predicted_rul_cycles": 3.0, "raw_predicted_rul_cycles": 3.0},
    ]
