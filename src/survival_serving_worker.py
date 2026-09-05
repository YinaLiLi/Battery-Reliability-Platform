"""Bounded x86-compatible Survival serving worker; it never starts Spark."""
import argparse
import json
import os
import time
from pathlib import Path

try:
    from .feature_contract import RUL_FEATURES
    from .shared_features import load_current_feature_rows
    from .serving_status import current_stream_state_row, serving_status_row, upsert_current_stream_state, upsert_serving_status
    from .stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    from .survival_stream_inference import current_survival_rows
except ImportError:
    from feature_contract import RUL_FEATURES
    from shared_features import load_current_feature_rows
    from serving_status import current_stream_state_row, serving_status_row, upsert_current_stream_state, upsert_serving_status
    from stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    from survival_stream_inference import current_survival_rows


def newest_finalized_features(features, boundary, benchmark_battery_ids=()):
    allowed = {(row["dataset"], row["battery_id"], row["cycle_index"]) for row in boundary["finalized_cycle_keys"]}
    latest = {}
    for row in features:
        key = (row.get("dataset"), row.get("battery_id"), row.get("cycle_index"))
        if key not in allowed or row["battery_id"] in benchmark_battery_ids:
            continue
        prior = latest.get(row["battery_id"])
        if prior is None or (row["cycle_index"], row.get("replay_sequence", 0)) >= (prior["cycle_index"], prior.get("replay_sequence", 0)):
            latest[row["battery_id"]] = row
    return [latest[battery] for battery in sorted(latest)]


def should_process(status):
    return not status or status.get("status") != "served"


def _selection(cursor):
    cursor.execute("""
        SELECT current.model_version, current.model_fingerprint AS selected_fingerprint,
               current.selection_revision, evaluation.model_fingerprint, evaluation.training_metadata
        FROM analytics.current_survival_models AS current
        JOIN analytics.survival_model_evaluations AS evaluation USING (model_version)
    """)
    row = cursor.fetchone()
    return dict(row) if row else None


def _status(cursor, state_id, selection):
    cursor.execute("""
        SELECT status FROM analytics.stream_serving_status
        WHERE dataset = 'MATR' AND state_id = %s AND consumer = 'survival_current'
          AND selection_revision = %s
    """, (state_id, selection["selection_revision"]))
    row = cursor.fetchone()
    return dict(row) if row else None


def _merge_predictions(cursor, rows):
    cursor.executemany("""
        INSERT INTO analytics.battery_current_survival_predictions
            (dataset, battery_id, cycle_index, horizon_cycles, survival_probability, model_version, model_fingerprint, state_id, replay_sequence, feature_contract_version, selection_revision, inference_created_at)
        VALUES (%(dataset)s, %(battery_id)s, %(cycle_index)s, %(horizon_cycles)s, %(survival_probability)s, %(model_version)s, %(model_fingerprint)s, %(state_id)s, %(replay_sequence)s, %(feature_contract_version)s, %(selection_revision)s, %(inference_created_at)s)
        ON CONFLICT (dataset, battery_id, horizon_cycles) DO UPDATE SET
            cycle_index = EXCLUDED.cycle_index, survival_probability = EXCLUDED.survival_probability,
            model_version = EXCLUDED.model_version, model_fingerprint = EXCLUDED.model_fingerprint,
            state_id = EXCLUDED.state_id, replay_sequence = EXCLUDED.replay_sequence,
            feature_contract_version = EXCLUDED.feature_contract_version,
            selection_revision = EXCLUDED.selection_revision, inference_created_at = EXCLUDED.inference_created_at
        WHERE EXCLUDED.replay_sequence > analytics.battery_current_survival_predictions.replay_sequence
           OR (EXCLUDED.replay_sequence = analytics.battery_current_survival_predictions.replay_sequence
               AND EXCLUDED.state_id = analytics.battery_current_survival_predictions.state_id
               AND EXCLUDED.selection_revision > analytics.battery_current_survival_predictions.selection_revision)
    """, rows)


def process_once(root, connection):
    root = Path(root)
    manifest = json.loads((root / "stream_state/latest.json").read_text())
    boundary = json.loads((root / manifest["finalized_cycle_boundary_ref"]).read_text())
    boundary = validate_finalized_cycle_boundary(boundary)
    validate_stream_state_manifest(manifest, boundary, expected_canonical_fingerprint=manifest["canonical_fingerprint"], expected_feature_contract_version=manifest["feature_contract_version"])
    with connection.cursor() as cursor:
        upsert_current_stream_state(cursor, current_stream_state_row("MATR", manifest))
        selection = _selection(cursor)
        if selection is None:
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", None))
            connection.commit()
            return {"status": "unavailable", "rows": 0}
        if not should_process(_status(cursor, manifest["state_id"], selection)):
            connection.commit()
            return {"status": "served", "rows": 0}
        try:
            if selection["selected_fingerprint"] != selection["model_fingerprint"]:
                raise RuntimeError("current Survival model fingerprint mismatch")
            metadata = selection["training_metadata"] if isinstance(selection["training_metadata"], dict) else json.loads(selection["training_metadata"])
            if metadata.get("feature_version", "").rsplit(":", 1)[-1] != manifest["feature_contract_version"].rsplit(":", 1)[-1]:
                raise RuntimeError("current Survival model feature contract mismatch")
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection))
            benchmark = json.loads((root / "fixed_offline_benchmark/v1/benchmark.json").read_text())
            excluded = set(benchmark["splits"]["validation"]["battery_ids"]) | set(benchmark["splits"]["test"]["battery_ids"])
            features = load_current_feature_rows(root, manifest, excluded_battery_ids=excluded)
            if not features:
                upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection, status="served"))
                connection.commit()
                return {"status": "served", "rows": 0}
            import joblib
            model = joblib.load(root / "survival_generations" / selection["model_fingerprint"] / "selected_model.joblib")
            rows = current_survival_rows(model, features, feature_columns=RUL_FEATURES,
                model_version=selection["model_version"], model_fingerprint=selection["model_fingerprint"],
                state_id=manifest["state_id"], feature_contract_version=manifest["feature_contract_version"],
                selection_revision=selection["selection_revision"])
            _merge_predictions(cursor, rows)
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection, status="served", rows_written=len(rows)))
            connection.commit()
            return {"status": "served", "rows": len(rows)}
        except Exception as error:
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection,
                status="failed", error_message=str(error)[:500]))
            connection.commit()
            return {"status": "failed", "rows": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/matr"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    import psycopg
    while True:
        try:
            with psycopg.connect(os.environ["DATABASE_URL"], row_factory=psycopg.rows.dict_row) as connection:
                print(process_once(args.root, connection), flush=True)
        except Exception as error:
            print(f"survival serving failed: {error}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
