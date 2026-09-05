import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.shared_features import (
    append_shared_feature_rows,
    finalize_shared_features,
    generation_for_timestamp,
    generation_id_for,
    load_shared_feature_rows,
    selected_rows_digest,
    stage_shared_features,
    validate_feature_outlet,
    validate_shared_features,
)
from src.stream_state import build_compact_finalized_cycle_boundary


def _state(tmp_path, state_id="state-1"):
    path = tmp_path / "stream_state" / state_id / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "state_id": state_id,
        "feature_contract_version": "feature-v1",
        "finalized_cycle_boundary_fingerprint": "boundary-v1",
        "cutoff_metadata": {"replay_cutoff": "2025-08-14T00:00:00+00:00"},
        "arrived_train_battery_ids": ["b1", "b2"],
        "observed_eol_train_battery_ids": ["b1"],
        "censored_train_battery_ids": ["b2"],
    }, sort_keys=True))
    return path


def test_generation_timestamps_use_canonical_progressive_v3_cutoffs():
    assert generation_for_timestamp(datetime(2024, 2, 9, tzinfo=timezone.utc)) == "1.0"
    assert generation_for_timestamp(datetime(2024, 2, 10, tzinfo=timezone.utc)) == "1.1"
    assert generation_for_timestamp(datetime(2028, 3, 19, tzinfo=timezone.utc)) == "1.1"
    assert generation_for_timestamp(datetime(2028, 3, 20, tzinfo=timezone.utc)) == "1.2"
    assert generation_for_timestamp(datetime(2035, 4, 13, tzinfo=timezone.utc)) == "1.3"


def _parquet(path, value=1.0):
    path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"battery_id": "b1", "cycle_index": 1, "feature": value}]), path / "part-00000.parquet")


def test_shared_features_publish_once_and_reuse_only_exact_content(tmp_path):
    state = _state(tmp_path)
    target = tmp_path / "historical_features" / "state-1"
    staged = stage_shared_features(target)
    _parquet(staged)

    metadata = finalize_shared_features(staged, target, state_manifest_path=state, generation="1.2")

    assert metadata["state_id"] == "state-1"
    assert metadata["generation"] == "1.2"
    assert metadata["row_count"] == 1
    assert metadata["schema_fingerprint"]
    assert metadata["content_fingerprint"]
    assert metadata["source_state_fingerprint"]
    assert validate_shared_features(target, state_manifest_path=state, generation="1.2") == metadata

    retry = stage_shared_features(target)
    _parquet(retry)
    assert finalize_shared_features(retry, target, state_manifest_path=state, generation="1.2") == metadata

    conflict = stage_shared_features(target)
    _parquet(conflict, value=2.0)
    with pytest.raises(ValueError, match="different content"):
        finalize_shared_features(conflict, target, state_manifest_path=state, generation="1.2")


def test_shared_features_reject_partial_and_mutated_outputs(tmp_path):
    state = _state(tmp_path)
    partial = tmp_path / "historical_features" / "state-1"
    _parquet(partial)
    with pytest.raises(ValueError, match="incomplete"):
        validate_shared_features(partial, state_manifest_path=state, generation="1.2")

    partial.rename(tmp_path / "partial")
    staged = stage_shared_features(partial)
    _parquet(staged)
    finalize_shared_features(staged, partial, state_manifest_path=state, generation="1.2")
    pq.write_table(pa.Table.from_pylist([{"battery_id": "b1", "cycle_index": 1, "feature": 9.0}]), partial / "part-00000.parquet")
    with pytest.raises(ValueError, match="content fingerprint"):
        validate_shared_features(partial, state_manifest_path=state, generation="1.2")


def _cycle_rows(root):
    cycles = root / "cycle_summary"
    cycles.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "discharge_capacity_in_Ah": 1.0},
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 2, "discharge_capacity_in_Ah": 0.9},
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 3, "discharge_capacity_in_Ah": 0.8},
    ]), cycles / "part.parquet")


def test_persistent_outlet_appends_each_key_once_and_selects_cumulative_generations(tmp_path):
    root = tmp_path / "matr"
    _cycle_rows(root)
    outlet = root / "shared_feature_outlet"
    common = {"feature_contract_version": "features-v1", "canonical_source_fingerprint": "canonical-v1"}

    first = append_shared_feature_rows(outlet, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 10.0},
    ], generation="1.0", **common)
    second = append_shared_feature_rows(outlet, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 2, "derived": 20.0},
    ], generation="1.1", **common)
    retry = append_shared_feature_rows(outlet, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 10.0},
    ], generation="1.0", **common)

    assert first["appended_row_count"] == second["appended_row_count"] == 1
    assert retry["appended_row_count"] == 0
    assert len(list((outlet / "segments").glob("*.parquet"))) == 2
    assert [row["cycle_index"] for row in load_shared_feature_rows(root, generation_id=1)] == [1]
    rows = load_shared_feature_rows(root, generation_id=2)
    assert [row["cycle_index"] for row in rows] == [1, 2]
    assert rows[0]["discharge_capacity_in_Ah"] == 1.0
    assert rows[0]["generation_id"] == 1
    assert generation_id_for("1.3") == 4
    assert validate_feature_outlet(outlet)["row_count"] == 2


def test_persistent_outlet_rejects_overwrite_duplicate_canonical_columns_and_lineage_change(tmp_path):
    outlet = tmp_path / "shared_feature_outlet"
    common = {"feature_contract_version": "features-v1", "canonical_source_fingerprint": "canonical-v1"}
    append_shared_feature_rows(outlet, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 10.0},
    ], generation="1.0", **common)

    with pytest.raises(ValueError, match="immutable shared feature row"):
        append_shared_feature_rows(outlet, [
            {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 99.0},
        ], generation="1.0", **common)
    with pytest.raises(ValueError, match="immutable shared feature row"):
        append_shared_feature_rows(outlet, [
            {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 10.0},
        ], generation="1.1", **common)
    with pytest.raises(ValueError, match="canonical cycle columns"):
        append_shared_feature_rows(outlet, [
            {"dataset": "MATR", "battery_id": "b1", "cycle_index": 2, "soh": 0.9},
        ], generation="1.1", **common)
    with pytest.raises(ValueError, match="canonical source fingerprint"):
        append_shared_feature_rows(outlet, [
            {"dataset": "MATR", "battery_id": "b1", "cycle_index": 2, "derived": 20.0},
        ], generation="1.1", feature_contract_version="features-v1", canonical_source_fingerprint="canonical-v2")


def test_selected_row_digest_is_layout_independent_and_boundary_limited(tmp_path):
    root = tmp_path / "matr"
    _cycle_rows(root)
    outlet = root / "shared_feature_outlet"
    append_shared_feature_rows(outlet, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 10.0},
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 2, "derived": 20.0},
    ], generation="1.0", feature_contract_version="features-v1", canonical_source_fingerprint="canonical-v1")
    canonical = [{"dataset": "MATR", "battery_id": "b1", "cycle_index": cycle} for cycle in (1, 2, 3)]
    boundary = build_compact_finalized_cycle_boundary(
        [{**canonical[0], "replay_sequence": 7}], canonical_cycle_keys=canonical,
        canonical_fingerprint="canonical-v1", arrival_manifest_fingerprint="arrival-v1",
        feature_contract_version="features-v1",
    )

    rows = load_shared_feature_rows(root, generation_id=1, boundary=boundary)

    assert [row["cycle_index"] for row in rows] == [1]
    assert selected_rows_digest(rows) == selected_rows_digest(list(reversed(rows)))


def test_append_retry_recovers_an_identical_unpublished_segment(tmp_path):
    outlet = tmp_path / "shared_feature_outlet"
    rows = [{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 10.0}]
    common = {"generation": "1.0", "feature_contract_version": "features-v1", "canonical_source_fingerprint": "canonical-v1"}
    append_shared_feature_rows(outlet, rows, **common)
    receipt = next((outlet / "appends").glob("*.json"))
    receipt.unlink()

    with pytest.raises(ValueError, match="unpublished segment"):
        validate_feature_outlet(outlet)
    assert append_shared_feature_rows(outlet, rows, **common)["appended_row_count"] == 1
    assert validate_feature_outlet(outlet)["row_count"] == 1
