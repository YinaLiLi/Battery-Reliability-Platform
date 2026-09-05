from src.survival_stream_inference import current_survival_rows


class Model:
    def predict_survival_function(self, matrix):
        return [lambda horizon: 1.2 if horizon == 0 else 1.0 - horizon / 100 for _ in matrix]


def test_survival_rows_clamp_and_include_required_horizons():
    rows = current_survival_rows(Model(), [{"dataset":"MATR","battery_id":"b","cycle_index":3,"replay_sequence":5,"x":1}], feature_columns=("x",), model_version="m", model_fingerprint="f", state_id="s", feature_contract_version="v", selection_revision=1)
    assert {row["horizon_cycles"] for row in rows} >= {0, 50, 100, 200}
    assert rows[0]["survival_probability"] == 1.0
    assert all(left["survival_probability"] >= right["survival_probability"] for left, right in zip(rows, rows[1:]))
