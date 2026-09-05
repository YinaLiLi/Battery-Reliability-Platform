from src import survival_serving_worker as worker
from src.shared_features import append_shared_feature_rows
from src.spark_streaming import _latest_features
from src.stream_runtime import publish_state_artifacts
from src.survival_models import FEATURE_VERSION


class Connection:
    def __init__(self): self.committed = False
    def cursor(self): return self
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def commit(self): self.committed = True


def test_no_current_survival_model_records_unavailable_without_blocking_finalized_state(monkeypatch, tmp_path):
    manifest = publish_state_artifacts(
        tmp_path, finalized_keys=[{"dataset": "MATR", "battery_id": "b", "cycle_index": 1}],
        state_rows=[{"dataset": "MATR", "battery_id": "b", "cycle_index": 1}],
        feature_rows=[{"dataset": "MATR", "battery_id": "b", "cycle_index": 1}],
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival", feature_contract_version="features",
        eligible_completed_training_batteries=[], cutoff_metadata={}, kafka_offsets={},
    )
    statuses, states = [], []
    monkeypatch.setattr(worker, "_selection", lambda _: None)
    monkeypatch.setattr(worker, "upsert_serving_status", lambda _, row: statuses.append(row))
    monkeypatch.setattr(worker, "upsert_current_stream_state", lambda _, row: states.append(row))

    connection = Connection()
    assert worker.process_once(tmp_path, connection) == {"status": "unavailable", "rows": 0}
    assert connection.committed and states[0]["state_id"] == manifest["state_id"]
    assert statuses[0]["status"] == "unavailable"


def test_configured_survival_failure_is_recorded_without_retracting_state(monkeypatch, tmp_path):
    manifest = publish_state_artifacts(
        tmp_path, finalized_keys=[{"dataset": "MATR", "battery_id": "b", "cycle_index": 1}],
        state_rows=[{"dataset": "MATR", "battery_id": "b", "cycle_index": 1}],
        feature_rows=[{"dataset": "MATR", "battery_id": "b", "cycle_index": 1}],
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival", feature_contract_version="features",
        eligible_completed_training_batteries=[], cutoff_metadata={}, kafka_offsets={},
    )
    selection = {"model_version": "v", "model_fingerprint": "f", "selected_fingerprint": "f", "selection_revision": 1,
                 "training_metadata": {"feature_version": "survival-landmark-features:wrong"}}
    statuses = []
    monkeypatch.setattr(worker, "_selection", lambda _: selection)
    monkeypatch.setattr(worker, "_status", lambda *_: None)
    monkeypatch.setattr(worker, "upsert_serving_status", lambda _, row: statuses.append(row))
    monkeypatch.setattr(worker, "upsert_current_stream_state", lambda *_: None)

    assert worker.process_once(tmp_path, Connection()) == {"status": "failed", "rows": 0}
    assert statuses[-1]["status"] == "failed"
    assert statuses[-1]["state_id"] == manifest["state_id"]


def test_current_rul_and_survival_read_the_same_finalized_feature_outlet(monkeypatch, tmp_path):
    feature_rows = [
        {"dataset": "MATR", "battery_id": "b", "cycle_index": 1, "replay_sequence": 1},
        {"dataset": "MATR", "battery_id": "b", "cycle_index": 2, "replay_sequence": 2},
    ]
    manifest = publish_state_artifacts(
        tmp_path,
        finalized_keys=[{key: row[key] for key in ("dataset", "battery_id", "cycle_index")} for row in feature_rows],
        state_rows=feature_rows, feature_rows=feature_rows,
        canonical_fingerprint="canonical", arrival_manifest_fingerprint="arrival",
        feature_contract_version="degradation-features:" + FEATURE_VERSION.rsplit(":", 1)[-1],
        eligible_completed_training_batteries=[], cutoff_metadata={}, kafka_offsets={},
    )
    cycles = tmp_path / "cycle_summary"
    cycles.mkdir()
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist([
        {"dataset": "MATR", "battery_id": "b", "cycle_index": 1},
        {"dataset": "MATR", "battery_id": "b", "cycle_index": 2},
    ]), cycles / "part.parquet")
    append_shared_feature_rows(tmp_path / "shared_feature_outlet", [
        {"dataset": "MATR", "battery_id": "b", "cycle_index": 1, "derived": 1.0},
        {"dataset": "MATR", "battery_id": "b", "cycle_index": 2, "derived": 2.0},
    ], generation="1.0", feature_contract_version=manifest["feature_contract_version"],
       canonical_source_fingerprint="canonical")
    benchmark = tmp_path / "fixed_offline_benchmark" / "v1" / "benchmark.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"splits":{"validation":{"battery_ids":[]},"test":{"battery_ids":[]}}}')
    seen = []
    selection = {"model_version": "v", "model_fingerprint": "f", "selected_fingerprint": "f", "selection_revision": 1,
                 "training_metadata": {"feature_version": FEATURE_VERSION}}
    monkeypatch.setattr(worker, "_selection", lambda _: selection)
    monkeypatch.setattr(worker, "_status", lambda *_: None)
    monkeypatch.setattr(worker, "upsert_current_stream_state", lambda *_: None)
    monkeypatch.setattr(worker, "upsert_serving_status", lambda *_: None)
    monkeypatch.setattr(worker, "current_survival_rows", lambda _model, rows, **_kwargs: seen.extend(rows) or [])
    monkeypatch.setattr(worker, "_merge_predictions", lambda *_: None)
    import joblib
    monkeypatch.setattr(joblib, "load", lambda _: object())

    assert worker.process_once(tmp_path, Connection())["status"] == "served"
    assert seen == _latest_features(tmp_path, manifest)
