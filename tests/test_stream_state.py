import pytest

from src.stream_state import (
    StreamStateValidationError,
    build_compact_finalized_cycle_boundary,
    build_finalized_cycle_boundary,
    create_stream_state_manifest,
    select_finalized_cycle_rows,
    validate_stream_state_manifest,
)


def _boundary():
    return build_finalized_cycle_boundary(
        [
            {"dataset": "MATR", "battery_id": "battery-b", "cycle_index": 2},
            {"dataset": "MATR", "battery_id": "battery-a", "cycle_index": 1},
        ],
        canonical_fingerprint="canonical-v1",
        arrival_manifest_fingerprint="arrival-v1",
        feature_contract_version="rul-causal-v1",
    )


def _manifest(boundary):
    return create_stream_state_manifest(
        boundary,
        boundary_ref="finalized_cycle_boundary/state-123",
        eligible_completed_training_batteries=["battery-b", "battery-a", "battery-a"],
        cutoff_metadata={"replay_event_time": "2020-01-02T00:00:00+00:00"},
        kafka_offsets={"battery_measurements": {"0": 8}, "battery_lifecycle": {"1": 3}},
    )


def test_stream_state_id_and_boundary_are_deterministic_for_equivalent_keys():
    first_boundary = _boundary()
    second_boundary = build_finalized_cycle_boundary(
        list(reversed(_boundary()["finalized_cycle_keys"])),
        canonical_fingerprint="canonical-v1",
        arrival_manifest_fingerprint="arrival-v1",
        feature_contract_version="rul-causal-v1",
    )

    first = _manifest(first_boundary)
    second = _manifest(second_boundary)

    assert first_boundary == second_boundary
    assert first["state_id"] == second["state_id"]
    assert first["state_id"].startswith("stream-state-")
    assert first["eligible_completed_training_batteries"] == ["battery-a", "battery-b"]
    assert first["finalized_cycle_boundary_ref"] == "finalized_cycle_boundary/state-123"


def test_manifest_validation_rejects_absent_or_mismatched_boundary_and_canonical_fingerprint():
    boundary = _boundary()
    manifest = _manifest(boundary)

    with pytest.raises(StreamStateValidationError, match="boundary is required"):
        validate_stream_state_manifest(manifest, None, expected_canonical_fingerprint="canonical-v1")

    incompatible_boundary = build_finalized_cycle_boundary(
        boundary["finalized_cycle_keys"],
        canonical_fingerprint="canonical-v2",
        arrival_manifest_fingerprint="arrival-v1",
        feature_contract_version="rul-causal-v1",
    )
    with pytest.raises(StreamStateValidationError, match="canonical fingerprint"):
        validate_stream_state_manifest(manifest, incompatible_boundary, expected_canonical_fingerprint="canonical-v1")

    with pytest.raises(StreamStateValidationError, match="canonical fingerprint"):
        validate_stream_state_manifest(manifest, boundary, expected_canonical_fingerprint="canonical-v2")

    other_boundary = build_finalized_cycle_boundary(
        [{"dataset": "MATR", "battery_id": "battery-a", "cycle_index": 2}],
        canonical_fingerprint="canonical-v1",
        arrival_manifest_fingerprint="arrival-v1",
        feature_contract_version="rul-causal-v1",
    )
    with pytest.raises(StreamStateValidationError, match="boundary fingerprint"):
        validate_stream_state_manifest(manifest, other_boundary, expected_canonical_fingerprint="canonical-v1")


def test_selection_uses_only_authoritative_finalized_cycle_keys():
    selected = select_finalized_cycle_rows(
        [
            {"dataset": "MATR", "battery_id": "battery-a", "cycle_index": 1, "value": "allowed"},
            {"dataset": "MATR", "battery_id": "battery-a", "cycle_index": 2, "value": "future"},
            {"dataset": "MATR", "battery_id": "battery-b", "cycle_index": 2, "value": "allowed"},
        ],
        _boundary(),
    )

    assert [row["value"] for row in selected] == ["allowed", "allowed"]


def test_manifest_rejects_noncanonical_or_invalid_kafka_offsets():
    boundary = _boundary()
    manifest = _manifest(boundary)
    manifest["kafka_offsets"] = {"battery_measurements": {"01": 8}}
    with pytest.raises(StreamStateValidationError, match="canonical order"):
        validate_stream_state_manifest(manifest, boundary, expected_canonical_fingerprint="canonical-v1")

    with pytest.raises(StreamStateValidationError, match="non-negative"):
        create_stream_state_manifest(boundary, boundary_ref="boundary", eligible_completed_training_batteries=[], cutoff_metadata={}, kafka_offsets={"battery_measurements": {"0": -1}})


def test_compact_boundary_requires_a_canonical_prefix_and_selects_the_same_rows():
    canonical = [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": cycle}
        for cycle in (1, 2, 3)
    ]
    finalized = [
        {**row, "replay_sequence": row["cycle_index"] + 10}
        for row in canonical[:2]
    ]

    boundary = build_compact_finalized_cycle_boundary(
        finalized, canonical_cycle_keys=canonical,
        canonical_fingerprint="canonical-v1", arrival_manifest_fingerprint="arrival-v1",
        feature_contract_version="features-v1",
    )
    manifest = _manifest(boundary)

    assert boundary["schema_version"] == "finalized-cycle-boundary-v2"
    assert boundary["finalized_cycle_ranges"] == [{
        "dataset": "MATR", "battery_id": "b1", "max_finalized_cycle_index": 2,
        "finalized_cycle_count": 2, "replay_sequence": 12,
    }]
    assert manifest["finalized_cycle_count"] == 2
    selected = select_finalized_cycle_rows([*canonical, {"dataset": "MATR", "battery_id": "b2", "cycle_index": 1}], boundary)
    assert [row["cycle_index"] for row in selected] == [1, 2]

    with pytest.raises(StreamStateValidationError, match="prefix-complete"):
        build_compact_finalized_cycle_boundary(
            [finalized[1]], canonical_cycle_keys=canonical,
            canonical_fingerprint="canonical-v1", arrival_manifest_fingerprint="arrival-v1",
            feature_contract_version="features-v1",
        )
