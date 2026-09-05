import pytest

from src.feature_contract import aggregate_cycle_samples, feature_rows


def test_causal_contract_computes_prior_window_slope_and_ignores_future_cycles():
    rows = [
        aggregate_cycle_samples([{
            "dataset": "MATR", "battery_id": "b", "cycle_index": cycle, "replay_sequence": cycle,
            "source_time_in_s": 100 + cycle, "voltage_in_V": 3.0 + cycle / 100,
            "current_in_A": -4.0, "temperature_in_C": 20 + cycle,
            "charge_capacity_in_Ah": 1.1, "discharge_capacity_in_Ah": 1.0 - cycle / 100,
            "internal_resistance_in_ohm": 0.01,
        }])
        for cycle in range(1, 12)
    ]
    features = feature_rows(rows)

    tenth = next(row for row in features if row["cycle_index"] == 10)
    assert tenth["rolling_capacity_mean_10"] == pytest.approx(sum(1.0 - cycle / 100 for cycle in range(1, 10)) / 9)
    assert tenth["capacity_slope_10"] == pytest.approx(-0.01)
    assert tenth["charge_time_delta"] == pytest.approx(1.0)

    extended = feature_rows([*rows, {**rows[-1], "cycle_index": 12, "discharge_capacity_in_Ah": 0.1}])
    assert next(row for row in extended if row["cycle_index"] == 10) == tenth
