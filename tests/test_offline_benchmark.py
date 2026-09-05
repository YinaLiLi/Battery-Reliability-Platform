import pytest

from src.offline_benchmark import BenchmarkValidationError, build_fixed_benchmark, select_benchmark_rows


def _manifest():
    return [
        {"battery_id": "train", "lineage_group_id": "lineage-train", "split": "train"},
        {"battery_id": "validation", "lineage_group_id": "lineage-validation", "split": "validation"},
        {"battery_id": "test", "lineage_group_id": "lineage-test", "split": "test"},
    ]


def test_fixed_benchmark_keeps_complete_held_out_rows_and_ids_stable():
    rows = [
        {"battery_id": "validation", "cycle_index": 1, "rul_cycles": 5},
        {"battery_id": "validation", "cycle_index": 2, "rul_cycles": 4},
        {"battery_id": "test", "cycle_index": 9, "rul_cycles": 1},
        {"battery_id": "train", "cycle_index": 1, "rul_cycles": 6},
    ]

    benchmark = build_fixed_benchmark(_manifest(), rows, feature_contract_version="rul-causal-v1")

    assert benchmark["benchmark_id"] == "fixed-offline-benchmark-v1"
    assert {key: benchmark["splits"]["validation"][key] for key in ("battery_ids", "lineage_group_ids", "row_count")} == {
        "battery_ids": ["validation"], "lineage_group_ids": ["lineage-validation"], "row_count": 2,
    }
    assert benchmark["splits"]["test"]["row_count"] == 1
    assert [row["battery_id"] for row in select_benchmark_rows(rows, benchmark, "validation")] == ["validation", "validation"]


def test_benchmark_rejects_a_manifest_that_moves_a_held_out_lineage():
    manifest = _manifest()
    benchmark = build_fixed_benchmark(manifest, [{"battery_id": "validation", "cycle_index": 1}], feature_contract_version="rul-causal-v1")
    moved = [dict(row) for row in manifest]
    moved[1]["lineage_group_id"] = "different-lineage"

    with pytest.raises(BenchmarkValidationError, match="lineage"):
        build_fixed_benchmark(moved, [{"battery_id": "validation", "cycle_index": 1}], feature_contract_version="rul-causal-v1", expected=benchmark)
