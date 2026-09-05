"""Small PostgreSQL status contract for finalized-state consumers."""
from datetime import datetime, timezone


def _now(value=None):
    return value or datetime.now(timezone.utc).isoformat()


def current_stream_state_row(dataset, manifest, published_at=None):
    return {"dataset": dataset, "state_id": manifest["state_id"],
            "feature_contract_version": manifest["feature_contract_version"], "published_at": _now(published_at)}


def serving_status_row(dataset, state_id, consumer, selection, *, status=None, rows_written=0, error_message=None, occurred_at=None):
    selection = selection or {}
    return {
        "dataset": dataset, "state_id": state_id, "consumer": consumer,
        "selection_revision": int(selection.get("selection_revision", 0)),
        "model_version": selection.get("model_version"), "model_fingerprint": selection.get("model_fingerprint"),
        "status": status or ("pending" if selection else "unavailable"), "rows_written": int(rows_written),
        "error_message": error_message, "updated_at": _now(occurred_at),
    }


def upsert_current_stream_state(cursor, row):
    cursor.execute("""
        INSERT INTO analytics.current_stream_states (dataset, state_id, feature_contract_version, published_at)
        VALUES (%(dataset)s, %(state_id)s, %(feature_contract_version)s, %(published_at)s)
        ON CONFLICT (dataset) DO UPDATE SET state_id = EXCLUDED.state_id,
            feature_contract_version = EXCLUDED.feature_contract_version, published_at = EXCLUDED.published_at
    """, row)


def upsert_serving_status(cursor, row):
    cursor.execute("""
        INSERT INTO analytics.stream_serving_status
            (dataset, state_id, consumer, selection_revision, model_version, model_fingerprint, status, rows_written, error_message, updated_at)
        VALUES (%(dataset)s, %(state_id)s, %(consumer)s, %(selection_revision)s, %(model_version)s, %(model_fingerprint)s, %(status)s, %(rows_written)s, %(error_message)s, %(updated_at)s)
        ON CONFLICT (dataset, state_id, consumer, selection_revision) DO UPDATE SET
            model_version = EXCLUDED.model_version, model_fingerprint = EXCLUDED.model_fingerprint,
            status = EXCLUDED.status, rows_written = EXCLUDED.rows_written,
            error_message = EXCLUDED.error_message, updated_at = EXCLUDED.updated_at
    """, row)
