from src.survival_serving_worker import newest_finalized_features, should_process


def test_worker_uses_only_boundary_rows_and_newest_nonbenchmark_cycle():
    boundary = {"finalized_cycle_keys": [
        {"dataset": "MATR", "battery_id": "train", "cycle_index": 1},
        {"dataset": "MATR", "battery_id": "train", "cycle_index": 3},
    ]}
    features = [
        {"dataset": "MATR", "battery_id": "train", "cycle_index": 1, "replay_sequence": 1},
        {"dataset": "MATR", "battery_id": "train", "cycle_index": 3, "replay_sequence": 3},
        {"dataset": "MATR", "battery_id": "benchmark", "cycle_index": 2, "replay_sequence": 2},
        {"dataset": "MATR", "battery_id": "train", "cycle_index": 4, "replay_sequence": 4},
    ]

    assert newest_finalized_features(features, boundary, {"benchmark"}) == [
        {"dataset": "MATR", "battery_id": "train", "cycle_index": 3, "replay_sequence": 3}
    ]


def test_worker_skips_an_already_served_state_and_selection_revision():
    assert not should_process({"status": "served"})
    assert should_process({"status": "failed"})
    assert should_process(None)
