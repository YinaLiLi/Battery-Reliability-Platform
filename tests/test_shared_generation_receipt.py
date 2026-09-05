import json

import pytest

from src.shared_generation_receipt import build_receipt, receipt_path, write_receipt


def test_shared_generation_receipt_is_immutable_and_state_bound(tmp_path):
    root = tmp_path / "matr"
    state_path = root / "stream_state" / "state-1" / "manifest.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "state_id": "state-1",
        "finalized_cycle_boundary_ref": "finalized_cycle_boundary/state-1/boundary.json",
        "finalized_cycle_boundary_fingerprint": "boundary-fingerprint",
        "feature_contract_version": "feature-v1",
        "arrival_manifest_fingerprint": "arrival-fingerprint",
        "arrived_train_battery_ids": ["b1", "b2"],
        "observed_eol_train_battery_ids": ["b1"],
        "censored_train_battery_ids": ["b2"],
        "cutoff_metadata": {"replay_cutoff": "2025-08-14T00:00:00+00:00"},
    }))
    benchmark = root / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"benchmark_id":"v1"}')

    receipt = build_receipt(state_path, "1.2", root=root)
    target = receipt_path(root, "state-1", "1.2")
    assert receipt["state_id"] == "state-1"
    assert receipt["training_features_path"] == str(root / "historical_features" / "state-1")
    assert receipt["benchmark_sha256"]
    assert write_receipt(target, receipt) == target
    assert write_receipt(target, receipt) == target
    with pytest.raises(ValueError, match="immutable"):
        write_receipt(target, {**receipt, "generation": "1.3"})
