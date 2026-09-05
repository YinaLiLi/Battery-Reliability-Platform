"""Immutable identity and publication for shared RUL/Survival features."""

from hashlib import sha256
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
from uuid import uuid4

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


METADATA_FILE = "_shared_features.json"
SCHEMA_VERSION = "shared-historical-features-v1"
OUTLET_SCHEMA_VERSION = "shared-feature-outlet-v1"
OUTLET_METADATA_FILE = "_outlet.json"
GENERATION_IDS = {"1.0": 1, "1.1": 2, "1.2": 3, "1.3": 4}
GENERATION_CUTOFFS = (
    ("1.0", "2024-02-09T00:00:00+00:00"),
    ("1.1", "2028-03-19T00:00:00+00:00"),
    ("1.2", "2032-04-27T00:00:00+00:00"),
    ("1.3", "2035-04-12T00:00:00+00:00"),
)
CANONICAL_V3_ROW_COUNTS = {"1.0": 14779, "1.1": 34313, "1.2": 52591, "1.3": 68436}
KEY_COLUMNS = ("dataset", "battery_id", "cycle_index")
CANONICAL_CYCLE_COLUMNS = {
    "eol_cycle", "rul_cycles", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "soh",
    "internal_resistance_in_ohm", "temperature_min_in_C", "temperature_max_in_C", "charge_time_in_s",
}


def _hash_bytes(value):
    return sha256(value).hexdigest()


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_id_for(generation):
    try:
        return GENERATION_IDS[str(generation)]
    except KeyError as error:
        raise ValueError(f"unsupported canonical generation: {generation}") from error


def generation_for_timestamp(value):
    """Return the first canonical model generation at or after row availability."""
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    for generation, cutoff in GENERATION_CUTOFFS:
        if timestamp <= datetime.fromisoformat(cutoff):
            return generation
    return GENERATION_CUTOFFS[-1][0]


def _row_key(row):
    try:
        key = row["dataset"], row["battery_id"], int(row["cycle_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shared feature rows require dataset, battery_id, and positive cycle_index") from error
    if not all(isinstance(value, str) and value for value in key[:2]) or key[2] < 1:
        raise ValueError("shared feature rows require dataset, battery_id, and positive cycle_index")
    return key


def _json_row(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)


def selected_rows_digest(rows):
    """Hash logical rows independently of Parquet files and ordering."""
    encoded = "\n".join(_json_row(dict(row)) for row in sorted(rows, key=_row_key))
    return _hash_bytes(encoded.encode())


def _append_receipts(path):
    directory = Path(path) / "appends"
    return [json.loads(receipt.read_text()) for receipt in sorted(directory.glob("*.json"))] if directory.is_dir() else []


def _outlet_rows(path, *, allow_orphans=False):
    path = Path(path)
    files = []
    for receipt in _append_receipts(path):
        if receipt.get("schema_version") != "shared-feature-append-v1":
            raise ValueError("unsupported shared feature append receipt")
        segment = path / receipt["segment"]
        if not segment.is_file():
            raise ValueError(f"shared feature append segment is missing: {segment}")
        segment_rows = ds.dataset([segment], format="parquet").to_table().to_pylist()
        if len(segment_rows) != receipt.get("row_count") or selected_rows_digest(segment_rows) != receipt.get("rows_sha256"):
            raise ValueError("shared feature append receipt does not match its segment")
        files.append(segment)
    orphaned = set((path / "segments").glob("*.parquet")) - set(files) if (path / "segments").is_dir() else set()
    if orphaned and not allow_orphans:
        raise ValueError("shared feature outlet contains an unpublished segment")
    return ds.dataset(files, format="parquet").to_table().to_pylist() if files else []


def _write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"immutable shared feature metadata differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_feature_outlet(path, *, feature_contract_version=None, canonical_source_fingerprint=None):
    """Validate one persistent outlet and its immutable keyed rows."""
    path = Path(path)
    metadata_path = path / OUTLET_METADATA_FILE
    if not metadata_path.is_file():
        raise ValueError(f"shared feature outlet is incomplete: {metadata_path} is missing")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != OUTLET_SCHEMA_VERSION:
        raise ValueError("unsupported shared feature outlet schema")
    for field, expected in (("feature_contract_version", feature_contract_version),
                            ("canonical_source_fingerprint", canonical_source_fingerprint)):
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(f"shared feature outlet {field.replace('_', ' ')} mismatch")
    rows = _outlet_rows(path)
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("shared feature outlet contains duplicate cycle keys")
    expected_columns = set(metadata.get("feature_columns", ())) | set(KEY_COLUMNS) | {"generation_id", "feature_contract_version"}
    for row in rows:
        if set(row) != expected_columns:
            raise ValueError("shared feature outlet row schema mismatch")
        if row["feature_contract_version"] != metadata["feature_contract_version"]:
            raise ValueError("shared feature outlet feature contract version mismatch")
        if isinstance(row["generation_id"], bool) or not isinstance(row["generation_id"], int) or row["generation_id"] < 1:
            raise ValueError("shared feature outlet generation_id must be a positive integer")
    return {**metadata, "row_count": len(rows), "selected_rows_sha256": selected_rows_digest(rows)}


def feature_outlet_rows(path):
    """Return normalized stored rows without joining canonical cycle facts."""
    validate_feature_outlet(path)
    return _outlet_rows(path)


def feature_outlet_key_maxima(path):
    """Read compact availability watermarks without scanning feature Parquet."""
    maxima = {}
    receipts = _append_receipts(path)
    if any("key_ranges" not in receipt for receipt in receipts):
        for row in feature_outlet_rows(path):
            key = row["dataset"], row["battery_id"]
            maxima[key] = max(maxima.get(key, 0), int(row["cycle_index"]))
        return maxima
    for receipt in receipts:
        for row in receipt["key_ranges"]:
            key = row["dataset"], row["battery_id"]
            maxima[key] = max(maxima.get(key, 0), int(row["max_cycle_index"]))
    return maxima


def _key_ranges(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["battery_id"]), []).append(int(row["cycle_index"]))
    return [
        {"dataset": key[0], "battery_id": key[1], "min_cycle_index": min(cycles),
         "max_cycle_index": max(cycles), "row_count": len(cycles)}
        for key, cycles in sorted(grouped.items())
    ]


def append_shared_feature_rows(path, rows, *, generation, feature_contract_version, canonical_source_fingerprint):
    """Append never-before-seen cycle keys; exact retries are no-ops."""
    path = Path(path)
    generation_id = generation_id_for(generation)
    incoming = [dict(row) for row in rows]
    if not incoming:
        raise ValueError("shared feature append requires at least one row")
    duplicate_columns = sorted(set().union(*(set(row) for row in incoming)) & CANONICAL_CYCLE_COLUMNS)
    if duplicate_columns:
        raise ValueError("shared feature rows must not duplicate canonical cycle columns: " + ", ".join(duplicate_columns))
    feature_columns = sorted(set(incoming[0]) - set(KEY_COLUMNS))
    if any(set(row) - set(KEY_COLUMNS) != set(feature_columns) for row in incoming):
        raise ValueError("shared feature append rows must use one schema")
    if "generation_id" in feature_columns or "feature_contract_version" in feature_columns:
        raise ValueError("generation_id and feature_contract_version are assigned by the outlet")
    normalized = []
    seen = set()
    for row in incoming:
        key = _row_key(row)
        if key in seen:
            raise ValueError("shared feature append contains duplicate cycle keys")
        seen.add(key)
        normalized.append({**row, "cycle_index": key[2], "generation_id": generation_id,
                           "feature_contract_version": feature_contract_version})
    metadata = {
        "schema_version": OUTLET_SCHEMA_VERSION,
        "canonical_source_fingerprint": canonical_source_fingerprint,
        "feature_contract_version": feature_contract_version,
        "feature_columns": feature_columns,
    }
    metadata_path = path / OUTLET_METADATA_FILE
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text())
        for field in ("schema_version", "canonical_source_fingerprint", "feature_contract_version", "feature_columns"):
            if existing_metadata.get(field) != metadata[field]:
                raise ValueError(f"shared feature outlet {field.replace('_', ' ')} mismatch")
    else:
        _write_once(metadata_path, json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    maxima = feature_outlet_key_maxima(path)
    overlaps = any(key[2] <= maxima.get(key[:2], 0) for key in seen)
    existing = ({_row_key(row): row for row in _outlet_rows(path, allow_orphans=True)} if overlaps else {})
    additions = []
    for row in normalized:
        prior = existing.get(_row_key(row))
        if prior is None:
            additions.append(row)
        elif prior != row:
            raise ValueError(f"immutable shared feature row differs for {_row_key(row)}")
    if not additions:
        return {**validate_feature_outlet(path), "appended_row_count": 0}
    additions.sort(key=_row_key)
    digest = selected_rows_digest(additions)
    segment = path / "segments" / f"{digest}.parquet"
    segment.parent.mkdir(parents=True, exist_ok=True)
    temporary = segment.with_name(segment.name + ".tmp")
    schema = pa.schema([
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("battery_id", pa.string(), nullable=False),
        pa.field("cycle_index", pa.int64(), nullable=False),
        pa.field("generation_id", pa.int64(), nullable=False),
        pa.field("feature_contract_version", pa.string(), nullable=False),
        *(pa.field(column, pa.float64()) for column in feature_columns),
    ])
    pq.write_table(pa.Table.from_pylist(additions, schema=schema), temporary)
    if segment.exists():
        segment_rows = pq.read_table(segment).to_pylist()
        if len(segment_rows) != len(additions) or selected_rows_digest(segment_rows) != digest:
            temporary.unlink(missing_ok=True)
            raise ValueError("unpublished shared feature segment conflicts with retry")
        temporary.unlink()
    else:
        os.replace(temporary, segment)
    receipt = {
        "schema_version": "shared-feature-append-v1", "generation_id": generation_id,
        "row_count": len(additions), "rows_sha256": digest, "segment": str(segment.relative_to(path)),
        "key_ranges": _key_ranges(additions),
    }
    _write_once(path / "appends" / f"{digest}.json", json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return {**metadata, "row_count": sum(item["row_count"] for item in _append_receipts(path)),
            "appended_row_count": len(additions)}


def seed_canonical_v3_feature_outlet(root):
    """Seed or validate the outlet from only canonical progressive-arrival-v3 states.

    This deliberately consumes the validated historical feature snapshots once; it
    never reads measurements and never writes a historical snapshot.
    """
    try:
        from .progressive_arrival import (
            GENERATION_ARRIVAL_COUNTS, RECORD_CLASS, SEMANTICS_VERSION,
        )
        from .feature_contract import SHARED_FEATURE_COLUMNS
        from .stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    except ImportError:
        from progressive_arrival import GENERATION_ARRIVAL_COUNTS, RECORD_CLASS, SEMANTICS_VERSION
        from feature_contract import SHARED_FEATURE_COLUMNS
        from stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest

    root = Path(root)
    expected_generations = tuple(GENERATION_IDS)
    states = {}
    for manifest_path in (root / "stream_state").glob("*/manifest.json"):
        state = json.loads(manifest_path.read_text())
        generation = str(state.get("generation") or state.get("cutoff_metadata", {}).get("generation", ""))
        if generation not in expected_generations:
            continue
        if state.get("generation_semantics_version") != SEMANTICS_VERSION:
            continue
        if state.get("record_class") != RECORD_CLASS:
            continue
        states.setdefault(generation, []).append((manifest_path, state))
    if set(states) != set(expected_generations) or any(len(items) != 1 for items in states.values()):
        raise ValueError("canonical progressive-arrival-v3 requires exactly one state for every generation")

    source_rows, boundaries, assignments, previous_keys = {}, {}, {}, set()
    feature_contract = canonical_fingerprint = None
    for generation in expected_generations:
        manifest_path, state = states[generation][0]
        cutoff = state.get("cutoff_metadata", {}).get("replay_cutoff")
        if cutoff != dict(GENERATION_CUTOFFS)[generation]:
            raise ValueError(f"canonical v3 generation {generation} has an unexpected replay cutoff")
        if len(state.get("arrived_train_battery_ids", ())) != GENERATION_ARRIVAL_COUNTS[generation]:
            raise ValueError(f"canonical v3 generation {generation} has an unexpected arrived cohort")
        boundary_path = root / state["finalized_cycle_boundary_ref"]
        boundary = json.loads(boundary_path.read_text())
        validate_stream_state_manifest(
            state, boundary, expected_canonical_fingerprint=state["canonical_fingerprint"],
            expected_arrival_manifest_fingerprint=state["arrival_manifest_fingerprint"],
            expected_feature_contract_version=state["feature_contract_version"],
        )
        validate_finalized_cycle_boundary(boundary)
        keys = {_row_key(row) for row in boundary.get("finalized_cycle_keys", ())}
        if len(keys) != CANONICAL_V3_ROW_COUNTS[generation] or not previous_keys.issubset(keys):
            raise ValueError(f"canonical v3 generation {generation} boundary is not the approved cumulative membership")
        snapshot = root / "historical_features" / state["state_id"]
        columns = [*KEY_COLUMNS, *SHARED_FEATURE_COLUMNS]
        rows = ds.dataset(snapshot, format="parquet").to_table(columns=columns).to_pylist()
        normalized = {_row_key(row): {column: row.get(column) for column in columns} for row in rows}
        if len(normalized) != len(rows) or set(normalized) != keys:
            raise ValueError(f"canonical v3 generation {generation} snapshot does not exactly match its boundary")
        for key in previous_keys:
            if source_rows[key] != normalized[key]:
                raise ValueError(f"canonical v3 feature values changed for finalized key {key}")
        source_rows.update(normalized)
        new_keys = keys - previous_keys
        for key in new_keys:
            assignments[key] = generation
        boundaries[generation] = boundary
        previous_keys = keys
        feature_contract = feature_contract or state["feature_contract_version"]
        canonical_fingerprint = canonical_fingerprint or state["canonical_fingerprint"]
        if (state["feature_contract_version"], state["canonical_fingerprint"]) != (feature_contract, canonical_fingerprint):
            raise ValueError("canonical v3 states disagree on feature or source identity")

    outlet = root / "shared_feature_outlet"
    if (outlet / OUTLET_METADATA_FILE).exists():
        existing = {_row_key(row): row for row in feature_outlet_rows(outlet)}
        for key, row in existing.items():
            generation = assignments.get(key)
            expected = {**source_rows.get(key, {}), "generation_id": generation_id_for(generation) if generation else None,
                        "feature_contract_version": feature_contract}
            if generation is None or row != expected:
                raise ValueError("existing shared feature outlet is not the approved canonical v3 seed")

    append_counts = {}
    for generation in expected_generations:
        new_rows = [source_rows[key] for key, assigned in assignments.items() if assigned == generation]
        result = append_shared_feature_rows(
            outlet, new_rows, generation=generation, feature_contract_version=feature_contract,
            canonical_source_fingerprint=canonical_fingerprint,
        )
        append_counts[generation] = result["appended_row_count"]

    outlet_rows = feature_outlet_rows(outlet)
    report = {}
    for generation in expected_generations:
        generation_id = generation_id_for(generation)
        selected = [row for row in outlet_rows if row["generation_id"] <= generation_id]
        expected_keys = {_row_key(row) for row in boundaries[generation]["finalized_cycle_keys"]}
        if {_row_key(row) for row in selected} != expected_keys:
            raise ValueError(f"shared feature outlet selection differs from canonical v3 generation {generation}")
        report[generation] = {
            "generation_id": generation_id,
            "new_row_count": sum(value == generation for value in assignments.values()),
            "appended_row_count": append_counts[generation],
            "cumulative_row_count": len(selected),
            "selected_rows_sha256": selected_rows_digest(selected),
            "state_id": states[generation][0][1]["state_id"],
            "boundary_fingerprint": boundaries[generation]["boundary_fingerprint"],
        }
    return {"outlet": str(outlet), "feature_contract_version": feature_contract,
            "canonical_source_fingerprint": canonical_fingerprint, "generations": report}


def load_shared_feature_rows(root, *, generation_id, boundary=None, battery_ids=None):
    """Return the canonical cycle facts joined to one cumulative feature selection."""
    root = Path(root)
    outlet = root / "shared_feature_outlet"
    validate_feature_outlet(outlet)
    rows = [row for row in _outlet_rows(outlet) if int(row["generation_id"]) <= int(generation_id)]
    if boundary is not None:
        try:
            from .stream_state import select_finalized_cycle_rows
        except ImportError:
            from stream_state import select_finalized_cycle_rows
        rows = select_finalized_cycle_rows(rows, boundary)
    allowed_batteries = set(battery_ids) if battery_ids is not None else None
    if allowed_batteries is not None:
        rows = [row for row in rows if row["battery_id"] in allowed_batteries]
    keys = {_row_key(row) for row in rows}
    cycles = ds.dataset(root / "cycle_summary", format="parquet").to_table().to_pylist()
    selected_cycles = [row for row in cycles if _row_key(row) in keys]
    cycle_by_key = {_row_key(row): row for row in selected_cycles}
    if len(selected_cycles) != len(cycle_by_key):
        raise ValueError("canonical cycle data contains duplicate shared feature keys")
    if set(cycle_by_key) != keys:
        raise ValueError("shared feature outlet references missing canonical cycle rows")
    return sorted(({**cycle_by_key[_row_key(row)], **row} for row in rows), key=_row_key)


def load_current_feature_rows(root, manifest, *, excluded_battery_ids=()):
    """Load each battery's newest finalized row from the shared outlet, with legacy fallback."""
    root = Path(root)
    boundary = json.loads((root / manifest["finalized_cycle_boundary_ref"]).read_text())
    excluded = set(excluded_battery_ids)
    if (root / "shared_feature_outlet" / OUTLET_METADATA_FILE).exists():
        rows = load_shared_feature_rows(root, generation_id=max(GENERATION_IDS.values()), boundary=boundary)
    else:
        rows = json.loads((root / "as_of_cycle_features" / manifest["state_id"] / "features.json").read_text())
        try:
            from .stream_state import select_finalized_cycle_rows
        except ImportError:
            from stream_state import select_finalized_cycle_rows
        rows = select_finalized_cycle_rows(rows, boundary)
    sequences = {(row["dataset"], row["battery_id"]): row["replay_sequence"]
                 for row in boundary.get("finalized_cycle_ranges", ())}
    latest = {}
    for row in rows:
        if row["battery_id"] in excluded:
            continue
        prior = latest.get(row["battery_id"])
        if prior is None or row["cycle_index"] >= prior["cycle_index"]:
            latest[row["battery_id"]] = dict(row)
    for row in latest.values():
        sequence = sequences.get((row["dataset"], row["battery_id"]))
        if sequence is not None:
            row["replay_sequence"] = sequence
    return [latest[battery] for battery in sorted(latest)]


def _cohort_checksums(state):
    return {
        name: _hash_bytes("\n".join(state.get(name, ())).encode())
        for name in ("arrived_train_battery_ids", "observed_eol_train_battery_ids", "censored_train_battery_ids")
    }


def _parquet_identity(path):
    files = sorted(Path(path).rglob("*.parquet"))
    if not files:
        raise ValueError("shared historical features contain no Parquet data")
    schemas = []
    row_count = 0
    content = []
    for file in files:
        parquet = pq.ParquetFile(file)
        row_count += parquet.metadata.num_rows
        schemas.append(json.dumps([
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ], sort_keys=True, separators=(",", ":")))
        content.append({"bytes": file.stat().st_size, "sha256": _hash_file(file)})
    if len(set(schemas)) != 1:
        raise ValueError("shared historical features contain inconsistent Parquet schemas")
    return {
        "row_count": row_count,
        "schema_fingerprint": _hash_bytes(schemas[0].encode()),
        "content_fingerprint": _hash_bytes(json.dumps(sorted(content, key=lambda item: (item["sha256"], item["bytes"])), separators=(",", ":")).encode()),
    }


def stage_shared_features(target):
    """Return an unpublished sibling path for a single Spark write."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f".{target.name}.{uuid4().hex}.partial")


def _metadata(dataset_path, state_manifest_path, generation):
    dataset_path, state_manifest_path = Path(dataset_path), Path(state_manifest_path)
    state = json.loads(state_manifest_path.read_text())
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "state_id": state["state_id"],
        "generation": str(generation),
        "feature_contract_version": state["feature_contract_version"],
        "cutoff_metadata": state["cutoff_metadata"],
        "cohort_checksums": _cohort_checksums(state),
        "source_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
        "source_state_fingerprint": _hash_file(state_manifest_path),
        **_parquet_identity(dataset_path),
    }


def validate_shared_features(path, *, state_manifest_path=None, generation=None, expected=None):
    """Validate completion, exact content, and state lineage for one shared dataset."""
    path = Path(path)
    metadata_path = path / METADATA_FILE
    if not metadata_path.is_file():
        raise ValueError(f"shared historical features are incomplete: {metadata_path} is missing")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shared historical feature metadata")
    if metadata.get("dataset_path") != str(path):
        raise ValueError("shared historical feature dataset path does not match its receipt")
    actual = _parquet_identity(path)
    for field, value in actual.items():
        if metadata.get(field) != value:
            raise ValueError(f"shared historical feature {field.replace('_', ' ')} mismatch")
    if state_manifest_path is not None:
        state = json.loads(Path(state_manifest_path).read_text())
        required = {
            "state_id": state["state_id"],
            "feature_contract_version": state["feature_contract_version"],
            "cutoff_metadata": state["cutoff_metadata"],
            "cohort_checksums": _cohort_checksums(state),
            "source_boundary_fingerprint": state["finalized_cycle_boundary_fingerprint"],
            "source_state_fingerprint": _hash_file(state_manifest_path),
        }
        if generation is not None:
            required["generation"] = str(generation)
        for field, value in required.items():
            if metadata.get(field) != value:
                raise ValueError(f"shared historical feature {field.replace('_', ' ')} mismatch")
    if expected is not None and metadata != expected:
        raise ValueError("shared historical feature metadata differs from the immutable receipt")
    return metadata


def finalize_shared_features(staged, target, *, state_manifest_path, generation):
    """Atomically publish a staged dataset, or validate an exact retry."""
    staged, target = Path(staged), Path(target)
    if not staged.is_dir():
        raise ValueError(f"staged shared historical features are missing: {staged}")
    metadata = _metadata(staged, state_manifest_path, generation)
    metadata["dataset_path"] = str(target)
    (staged / METADATA_FILE).write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    if target.exists():
        try:
            existing = validate_shared_features(target, state_manifest_path=state_manifest_path, generation=generation)
            if existing != metadata:
                raise ValueError(f"immutable shared historical features have different content: {target}")
            return existing
        finally:
            shutil.rmtree(staged, ignore_errors=True)
    try:
        os.rename(staged, target)
    except OSError:
        if not target.exists():
            raise
        try:
            existing = validate_shared_features(target, state_manifest_path=state_manifest_path, generation=generation)
            if existing != metadata:
                raise ValueError(f"immutable shared historical features have different content: {target}")
            return existing
        finally:
            shutil.rmtree(staged, ignore_errors=True)
    return validate_shared_features(target, state_manifest_path=state_manifest_path, generation=generation)
