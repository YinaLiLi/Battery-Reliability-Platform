import json

import pytest

from src.stream_runtime import publish_state_artifacts


def test_manifest_pointer_is_published_only_after_all_state_artifacts(tmp_path):
    manifest = publish_state_artifacts(
        tmp_path,
        finalized_keys=[{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1}],
        state_rows=[{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "replay_sequence": 3}],
        feature_rows=[{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "replay_sequence": 3}],
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival", feature_contract_version="rul-causal-v1",
        eligible_completed_training_batteries=["b1"], cutoff_metadata={"batch_id": 4}, kafka_offsets={},
    )

    state = manifest["state_id"]
    assert (tmp_path / "finalized_cycle_boundary" / state / "boundary.json").exists()
    assert not (tmp_path / "as_of_cycle_state" / state).exists()
    assert not (tmp_path / "as_of_cycle_features" / state).exists()
    assert json.loads((tmp_path / "stream_state" / "latest.json").read_text()) == manifest


def test_existing_state_rejects_conflicting_immutable_manifest_content(tmp_path):
    kwargs = dict(
        finalized_keys=[{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1}],
        state_rows=[{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1}],
        feature_rows=[{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1}],
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival", feature_contract_version="rul-causal-v1",
        eligible_completed_training_batteries=[], cutoff_metadata={"batch_id": 1}, kafka_offsets={"battery_measurements": {"0": 4}},
    )
    publish_state_artifacts(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="immutable state artifact differs"):
        publish_state_artifacts(tmp_path, **{**kwargs, "kafka_offsets": {"battery_measurements": {"0": 5}}})
