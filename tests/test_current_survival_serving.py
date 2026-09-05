from src import survival_serving_worker as worker
from src.stream_runtime import publish_state_artifacts


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
