import numpy as np

from src.parse_nasa_data import parse_cycles


def test_parse_cycles_uses_only_prior_impedance_and_first_valid_capacity():
    cycles = [
        {"type": "impedance", "ambient_temperature": 24, "data": {"Re": np.array([[0.01]]), "Rct": 0.02}},
        {
            "type": "discharge",
            "ambient_temperature": np.array([[25]]),
            "data": {
                "Capacity": np.array([[2.0]]),
                "Voltage_measured": np.array([4.0, 3.0]),
                "Current_measured": np.array([-1.0, -1.2]),
                "Temperature_measured": np.array([20.0, 30.0]),
            },
        },
        {"type": "impedance", "ambient_temperature": 25, "data": {"Re": 0.03, "Rct": 0.04}},
        {
            "type": "discharge",
            "ambient_temperature": 26,
            "data": {
                "Capacity": 1.5,
                "Voltage_measured": np.array([3.9, 2.9]),
                "Current_measured": np.array([-1.1, -1.3]),
                "Temperature_measured": np.array([21.0, 31.0]),
            },
        },
        {
            "type": "discharge",
            "ambient_temperature": 27,
            "data": {
                "Capacity": np.array([]),
                "Voltage_measured": np.array([3.8]),
                "Current_measured": np.array([-1.0]),
                "Temperature_measured": np.array([22.0]),
            },
        },
    ]

    rows = parse_cycles("B0005", cycles)

    assert [row["discharge_cycle_number"] for row in rows] == [1, 2, 3]
    assert [row["capacity_retention"] for row in rows[:2]] == [1.0, 0.75]
    assert rows[0]["Re"] == 0.01
    assert rows[1]["Rct"] == 0.04
    assert rows[2]["capacity"] is None
    assert rows[2]["capacity_retention"] is None
    assert rows[1]["avg_voltage"] == 3.4
    assert rows[0]["max_temperature"] == 30.0


def test_parse_cycles_keeps_impedance_missing_until_available():
    rows = parse_cycles(
        "B0006",
        [
            {
                "type": "discharge",
                "ambient_temperature": 24,
                "data": {
                    "Capacity": 2.0,
                    "Voltage_measured": np.array([3.5]),
                    "Current_measured": np.array([-1.0]),
                    "Temperature_measured": np.array([25.0]),
                },
            },
            {"type": "impedance", "ambient_temperature": 24, "data": {"Re": 0.01, "Rct": 0.02}},
        ],
    )

    assert rows[0]["Re"] is None
    assert rows[0]["Rct"] is None
