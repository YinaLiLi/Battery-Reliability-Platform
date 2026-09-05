import json
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.shared_features import append_shared_feature_rows, finalize_shared_features, stage_shared_features
from src.shared_generation_receipt import build_receipt, read_receipt, receipt_path, validate_state_manifest_path, write_receipt
from src.stream_state import build_finalized_cycle_boundary, create_stream_state_manifest
from src.survival_models import plan_from_receipt as survival_plan_from_receipt
from src.train_matr_models import plan_from_receipt as rul_plan_from_receipt


def _write_state(root, *, cutoff="2022-12-02T00:00:00+00:00", kafka_offsets=None):
    boundary = build_finalized_cycle_boundary(
        [{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1}],
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="schedule",
        feature_contract_version="feature-v1",
    )
    state_id = create_stream_state_manifest(
        boundary, boundary_ref="pending", eligible_completed_training_batteries=["b1"],
        cutoff_metadata={"replay_cutoff": cutoff}, kafka_offsets=kafka_offsets or {},
    )["state_id"]
    boundary_path = root / "finalized_cycle_boundary" / state_id / "boundary.json"
    boundary_path.parent.mkdir(parents=True)
    boundary_path.write_text(json.dumps(boundary, sort_keys=True))
    state = create_stream_state_manifest(
        boundary, boundary_ref=str(boundary_path.relative_to(root)),
        eligible_completed_training_batteries=["b1"], cutoff_metadata={"replay_cutoff": cutoff},
        kafka_offsets=kafka_offsets or {},
    )
    state = {**state, "arrived_train_battery_ids": ["b1"],
             "observed_eol_train_battery_ids": ["b1"], "censored_train_battery_ids": []}
    state_path = root / "stream_state" / state_id / "manifest.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state, sort_keys=True))
    return state_path, state


def test_shared_generation_receipt_is_immutable_and_state_bound(tmp_path):
    root = tmp_path / "matr"
    state_path, state = _write_state(root, cutoff="2025-08-14T00:00:00+00:00")
    benchmark = root / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"benchmark_id":"v1"}')
    cycles = root / "cycle_summary"
    cycles.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "rul_cycles": 2.0}]), cycles / "part.parquet")
    features = root / "shared_feature_outlet"
    append_shared_feature_rows(features, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 3.0},
    ], generation="1.2", feature_contract_version="feature-v1", canonical_source_fingerprint="canonical")

    receipt = build_receipt(state_path, "1.2", root=root)
    target = receipt_path(root, state["state_id"], "1.2")
    assert receipt["state_id"] == state["state_id"]
    assert receipt["schema_version"] == "shared-generation-receipt-v3"
    assert receipt["generation_id"] == 3
    assert receipt["shared_feature_outlet_ref"] == "shared_feature_outlet"
    assert receipt["selected_row_count"] == 1
    assert receipt["selected_rows_sha256"]
    assert receipt["cohort_checksum"]
    assert receipt["benchmark_sha256"]
    assert write_receipt(target, receipt) == target
    assert write_receipt(target, receipt) == target
    with pytest.raises(ValueError, match="immutable"):
        write_receipt(target, {**receipt, "generation": "1.3"})

    assert read_receipt(target)["selected_rows_sha256"] == receipt["selected_rows_sha256"]


def test_both_trainers_reject_a_mutated_or_substituted_shared_dataset(tmp_path):
    root = tmp_path / "matr"
    state_path, state = _write_state(root)
    benchmark = root / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"benchmark_id":"v1"}')
    cycles = root / "cycle_summary"
    cycles.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "rul_cycles": 2.0}]), cycles / "part.parquet")
    features = root / "shared_feature_outlet"
    append_shared_feature_rows(features, [
        {"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 3.0},
    ], generation="1.0", feature_contract_version="feature-v1", canonical_source_fingerprint="canonical")
    receipt_file = receipt_path(root, state["state_id"], "1.0")
    write_receipt(receipt_file, build_receipt(state_path, "1.0", root=root))
    manifest = [{"battery_id": "b1", "lineage_group_id": "l1", "split": "train", "start_time": "2020-01-01T00:00:00+00:00", "first_source_cycle": 1, "last_source_cycle": 3, "eol_cycle": 3, "valid_eol_label": True, "arrival_rank": 0, "schedule_fingerprint": "schedule"}]

    rul_plan = rul_plan_from_receipt(receipt_file, manifest, root=root)
    survival_plan = survival_plan_from_receipt(receipt_file, manifest, root=root)
    assert rul_plan["snapshot_id"] == survival_plan["snapshot_id"] == state["state_id"]
    assert rul_plan["shared_feature_rows"] == survival_plan["shared_feature_rows"]
    assert rul_plan["shared_feature_metadata"] == survival_plan["shared_feature_metadata"]

    original_receipt = json.loads(receipt_file.read_text())
    receipt_file.write_text(json.dumps({**original_receipt, "cutoff_metadata": {"replay_cutoff": "2099-01-01T00:00:00+00:00"}}))
    for loader in (rul_plan_from_receipt, survival_plan_from_receipt):
        with pytest.raises(ValueError, match="cutoff metadata"):
            loader(receipt_file, manifest, root=root)
    receipt_file.write_text(json.dumps(original_receipt))

    boundary_path = root / state["finalized_cycle_boundary_ref"]
    original_boundary = boundary_path.read_text()
    boundary_path.write_text('{"changed":true}')
    for loader in (rul_plan_from_receipt, survival_plan_from_receipt):
        with pytest.raises(ValueError, match="boundary"):
            loader(receipt_file, manifest, root=root)
    boundary_path.write_text(original_boundary)

    segment = next((features / "segments").glob("*.parquet"))
    original_segment = segment.read_bytes()
    pq.write_table(pa.Table.from_pylist([{
        "dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "generation_id": 1,
        "feature_contract_version": "feature-v1", "derived": 99.0,
    }]), segment)
    for loader in (rul_plan_from_receipt, survival_plan_from_receipt):
        with pytest.raises(ValueError, match="append receipt does not match"):
            loader(receipt_file, manifest, root=root)
    segment.write_bytes(original_segment)

    substituted = {**original_receipt, "shared_feature_outlet_ref": "replacement"}
    receipt_file.write_text(json.dumps(substituted, sort_keys=True))
    for loader in (rul_plan_from_receipt, survival_plan_from_receipt):
        with pytest.raises(ValueError, match="outlet ref"):
            loader(receipt_file, manifest, root=root)


def test_continuous_training_requires_streaming_lineage_and_complete_shared_cohort(tmp_path):
    root = tmp_path / "matr"
    boundary = build_finalized_cycle_boundary(
        [{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1}],
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        feature_contract_version="feature-v1",
    )
    boundary_path = root / "finalized_cycle_boundary" / "state" / "boundary.json"
    boundary_path.parent.mkdir(parents=True)
    boundary_path.write_text(json.dumps(boundary))

    def write_state(kafka_offsets, **extra):
        state = create_stream_state_manifest(
            boundary, boundary_ref=str(boundary_path.relative_to(root)),
            eligible_completed_training_batteries=["b1"],
            cutoff_metadata={"replay_cutoff": "2025-01-01T00:00:00+00:00"},
            kafka_offsets=kafka_offsets,
        )
        path = root / "stream_state" / state["state_id"] / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**state, **extra}))
        return path

    missing_offsets = write_state({}, arrived_train_battery_ids=["b1"], observed_eol_train_battery_ids=["b1"], censored_train_battery_ids=[])
    with pytest.raises(ValueError, match="Kafka offsets"):
        validate_state_manifest_path(missing_offsets, root=root, require_streaming=True)

    missing_cohort = write_state({"battery_measurements": {"0": 4}})
    with pytest.raises(ValueError, match="shared training cohort"):
        validate_state_manifest_path(missing_cohort, root=root, require_streaming=True)


def test_legacy_v2_receipt_and_snapshot_remain_readable(tmp_path):
    root = tmp_path / "matr"
    state_path, state = _write_state(root)
    benchmark = root / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"benchmark_id":"v1"}')
    target = root / "historical_features" / state["state_id"]
    staged = stage_shared_features(target)
    staged.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"dataset": "MATR", "battery_id": "b1", "cycle_index": 1, "derived": 3.0}]), staged / "part.parquet")
    metadata = finalize_shared_features(staged, target, state_manifest_path=state_path, generation="1.0")
    cohort_checksums = {
        name: sha256("\n".join(state.get(name, ())).encode()).hexdigest()
        for name in ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")
    }
    receipt = {
        "schema_version": "shared-generation-receipt-v2", "generation": "1.0",
        "state_id": state["state_id"], "state_manifest_path": str(state_path),
        "state_manifest_sha256": sha256(state_path.read_bytes()).hexdigest(),
        "finalized_cycle_boundary_ref": state["finalized_cycle_boundary_ref"],
        "finalized_cycle_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
        "feature_contract_version": state["feature_contract_version"],
        "arrival_manifest_fingerprint": state["arrival_manifest_fingerprint"],
        "cutoff_metadata": state["cutoff_metadata"], "cohort_checksums": cohort_checksums,
        "training_features_path": str(target), "training_features": metadata,
        "benchmark_path": str(benchmark), "benchmark_sha256": sha256(benchmark.read_bytes()).hexdigest(),
    }
    legacy = root / "shared_generation_receipts" / state["state_id"] / "1.0.json"
    write_receipt(legacy, receipt)

    assert read_receipt(legacy) == receipt
