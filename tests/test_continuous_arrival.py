from datetime import datetime, timezone

from src.continuous_arrival import (
    build_arrival_manifest,
    eligible_training_batteries,
    model_fingerprint,
    next_snapshot,
)


def _provenance():
    return [
        {"battery_id": "train-valid", "lineage_group_id": "lineage-a", "batch_id": "batch_1", "charge_policy": "p"},
        {"battery_id": "train-unverified", "lineage_group_id": "lineage-b", "batch_id": "batch_1", "charge_policy": "p"},
        {"battery_id": "validation-valid", "lineage_group_id": "lineage-c", "batch_id": "batch_2", "charge_policy": "p"},
        {"battery_id": "test-valid", "lineage_group_id": "lineage-d", "batch_id": "batch_2", "charge_policy": "p"},
    ]


def _cycles():
    return [
        {"battery_id": "train-valid", "cycle_index": 1, "eol_cycle": 2},
        {"battery_id": "train-valid", "cycle_index": 2, "eol_cycle": 2},
        {"battery_id": "train-unverified", "cycle_index": 1, "eol_cycle": 3},
        {"battery_id": "train-unverified", "cycle_index": 2, "eol_cycle": 3},
        {"battery_id": "validation-valid", "cycle_index": 1, "eol_cycle": 1},
        {"battery_id": "test-valid", "cycle_index": 1, "eol_cycle": 1},
    ]


def test_arrival_manifest_classifies_only_source_supported_eol_labels_as_valid():
    manifest = build_arrival_manifest(
        _provenance(),
        _cycles(),
        {"train": {"train-valid", "train-unverified"}, "validation": {"validation-valid"}, "test": {"test-valid"}},
        start_epoch=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    rows = {row["battery_id"]: row for row in manifest}

    assert rows["train-valid"]["label_status"] == "valid_observed_endpoint"
    assert rows["train-valid"]["valid_eol_label"]
    assert rows["train-unverified"]["label_status"] == "unverified_endpoint_after_source_end"
    assert not rows["train-unverified"]["valid_eol_label"]
    assert rows["train-unverified"]["eol_cycle_delta"] == 1
    assert sorted(row["arrival_rank"] for row in manifest) == list(range(4))
    assert len({row["schedule_fingerprint"] for row in manifest}) == 1


def test_training_eligibility_requires_completion_and_a_valid_observed_eol():
    manifest = [
        {"battery_id": "complete-valid", "split": "train", "valid_eol_label": True},
        {"battery_id": "complete-unverified", "split": "train", "valid_eol_label": False},
        {"battery_id": "not-complete", "split": "train", "valid_eol_label": True},
        {"battery_id": "validation", "split": "validation", "valid_eol_label": True},
    ]
    lifecycle = [
        {"battery_id": "complete-valid", "event_type": "eol_observed"},
        {"battery_id": "complete-valid", "event_type": "replay_complete"},
        {"battery_id": "complete-unverified", "event_type": "replay_complete"},
        {"battery_id": "not-complete", "event_type": "eol_observed"},
        {"battery_id": "validation", "event_type": "eol_observed"},
        {"battery_id": "validation", "event_type": "replay_complete"},
    ]

    assert eligible_training_batteries(manifest, lifecycle) == {"complete-valid"}
    assert eligible_training_batteries(manifest, [{"battery_id": "complete-valid", "eol_observed": True, "replay_complete": True}]) == {"complete-valid"}


def test_snapshot_thresholds_count_only_valid_eligible_train_batteries():
    eligible = [f"b{index}" for index in range(94)]

    assert next_snapshot(eligible, published_counts=set()) == (26, eligible[:26])
    assert next_snapshot(eligible, published_counts={26}) == (51, eligible[:51])
    assert next_snapshot(eligible, published_counts={26, 51, 76}) == (94, eligible[:94])
    assert next_snapshot(eligible, published_counts={26, 51, 76, 94}) is None


def test_model_fingerprint_is_stable_for_one_snapshot_and_changes_with_the_pool():
    common = {"manifest_fingerprint": "manifest", "split_version": "lineage-split-42", "feature_version": "features-v1", "model_config": {"seed": 42}}

    assert model_fingerprint(["b2", "b1"], **common) == model_fingerprint(["b1", "b2"], **common)
    assert model_fingerprint(["b1"], **common) != model_fingerprint(["b1", "b2"], **common)
