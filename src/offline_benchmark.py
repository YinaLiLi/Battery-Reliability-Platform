"""Immutable, complete held-out data for RUL generation comparison."""

from hashlib import sha256
import json


BENCHMARK_ID = "fixed-offline-benchmark-v1"


class BenchmarkValidationError(ValueError):
    """The fixed offline benchmark no longer matches its recorded cohort."""


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _split_members(manifest, split):
    rows = [row for row in manifest if row.get("split") == split]
    return {
        "battery_ids": sorted(row["battery_id"] for row in rows),
        "lineage_group_ids": sorted(row["lineage_group_id"] for row in rows),
    }


def select_benchmark_rows(rows, benchmark, split):
    """Return complete held-out rows without consulting a stream cutoff."""
    battery_ids = set(benchmark["splits"][split]["battery_ids"])
    return [row for row in rows if row.get("battery_id") in battery_ids]


def build_fixed_benchmark(manifest, rows, *, feature_contract_version, expected=None):
    """Record the fixed validation/test cohorts and their complete row hashes."""
    splits = {}
    for split in ("validation", "test"):
        members = _split_members(manifest, split)
        selected = [row for row in rows if row.get("battery_id") in set(members["battery_ids"])]
        splits[split] = {**members, "row_count": len(selected), "content_fingerprint": _digest(selected)}
    benchmark = {
        "benchmark_id": BENCHMARK_ID,
        "feature_contract_version": feature_contract_version,
        "splits": splits,
        "benchmark_fingerprint": _digest({"feature_contract_version": feature_contract_version, "splits": splits}),
    }
    if expected:
        for split in ("validation", "test"):
            for field in ("battery_ids", "lineage_group_ids", "row_count", "content_fingerprint"):
                if benchmark["splits"][split][field] != expected["splits"][split][field]:
                    raise BenchmarkValidationError(f"fixed {split} benchmark {field.replace('_', ' ')} changed")
    return benchmark
