"""Bounded x86-compatible Survival serving worker; it never starts Spark."""
import argparse
import json
import os
import time
from pathlib import Path

try:
    from .feature_contract import RUL_FEATURES
    from .shared_features import load_current_monitoring_feature_rows
    from .serving_status import current_stream_state_row, serving_status_row, upsert_current_stream_state, upsert_serving_status
    from .stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    from .survival_stream_inference import HORIZON_GRID, current_survival_rows
except ImportError:
    from feature_contract import RUL_FEATURES
    from shared_features import load_current_monitoring_feature_rows
    from serving_status import current_stream_state_row, serving_status_row, upsert_current_stream_state, upsert_serving_status
    from stream_state import validate_finalized_cycle_boundary, validate_stream_state_manifest
    from survival_stream_inference import HORIZON_GRID, current_survival_rows


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


def should_process(status, *, expected_rows, complete):
    return not complete or not status or status.get("status") != "served" or status.get("rows_written") != expected_rows


def coverage_complete(features, coverage):
    expected = {row["battery_id"]: int(row["cycle_index"]) for row in features}
    actual = {
        row["battery_id"]: (int(row["cycle_index"]), tuple(int(value) for value in row["horizons"]))
        for row in coverage
    }
    horizons = tuple(HORIZON_GRID)
    return len(actual) == len(coverage) == len(expected) and all(actual.get(battery) == (cycle, horizons) for battery, cycle in expected.items())


def _selection(cursor):
    cursor.execute("""
        SELECT current.dataset, current.model_version, current.model_fingerprint AS selected_fingerprint,
               current.selection_revision, evaluation.model_fingerprint, evaluation.training_metadata
        FROM analytics.current_survival_models AS current
        JOIN analytics.survival_model_evaluations AS evaluation USING (model_version)
    """)
    row = cursor.fetchone()
    return dict(row) if row else None


def _status(cursor, state_id, selection):
    cursor.execute("""
        SELECT status, rows_written FROM analytics.stream_serving_status
        WHERE dataset = 'MATR' AND state_id = %s AND consumer = 'survival_current'
          AND selection_revision = %s AND model_version = %s AND model_fingerprint = %s
    """, (state_id, selection["selection_revision"], selection["model_version"], selection["model_fingerprint"]))
    row = cursor.fetchone()
    return dict(row) if row else None


def _coverage(cursor, state_id, selection):
    cursor.execute("""
        SELECT battery_id, cycle_index, ARRAY_AGG(horizon_cycles ORDER BY horizon_cycles) AS horizons
        FROM analytics.battery_current_survival_predictions
        WHERE dataset = %s AND state_id = %s AND model_version = %s
          AND model_fingerprint = %s AND selection_revision = %s
        GROUP BY battery_id, cycle_index
    """, (selection["dataset"], state_id, selection["model_version"], selection["model_fingerprint"], selection["selection_revision"]))
    return [dict(row) for row in cursor.fetchall()]


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
        WHERE EXISTS (
                  SELECT 1
                  FROM analytics.current_stream_states AS state
                  JOIN analytics.current_survival_models AS current USING (dataset)
                  WHERE state.dataset = EXCLUDED.dataset AND state.state_id = EXCLUDED.state_id
                    AND current.model_version = EXCLUDED.model_version
                    AND current.model_fingerprint = EXCLUDED.model_fingerprint
                    AND current.selection_revision = EXCLUDED.selection_revision
              )
          AND (EXCLUDED.cycle_index > analytics.battery_current_survival_predictions.cycle_index
               OR (EXCLUDED.cycle_index = analytics.battery_current_survival_predictions.cycle_index
                   AND (EXCLUDED.replay_sequence > analytics.battery_current_survival_predictions.replay_sequence
                        OR (EXCLUDED.replay_sequence = analytics.battery_current_survival_predictions.replay_sequence
                            AND (EXCLUDED.selection_revision > analytics.battery_current_survival_predictions.selection_revision
                                 OR EXCLUDED.state_id IS DISTINCT FROM analytics.battery_current_survival_predictions.state_id)))))
    """, rows)


def prune_stale_predictions(cursor, features):
    """Keep only the newest finalized battery/cycle set for the current dataset."""
    expected = sorted((row["battery_id"], int(row["cycle_index"])) for row in features)
    cursor.execute("""
        DELETE FROM analytics.battery_current_survival_predictions AS prediction
        WHERE prediction.dataset = %s
          AND NOT EXISTS (
              SELECT 1 FROM unnest(%s::text[], %s::integer[]) AS expected(battery_id, cycle_index)
              WHERE expected.battery_id = prediction.battery_id
                AND expected.cycle_index = prediction.cycle_index
          )
    """, (features[0]["dataset"], [row[0] for row in expected], [row[1] for row in expected]))


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
        try:
            if selection["selected_fingerprint"] != selection["model_fingerprint"]:
                raise RuntimeError("current Survival model fingerprint mismatch")
            metadata = selection["training_metadata"] if isinstance(selection["training_metadata"], dict) else json.loads(selection["training_metadata"])
            if metadata.get("feature_version", "").rsplit(":", 1)[-1] != manifest["feature_contract_version"].rsplit(":", 1)[-1]:
                raise RuntimeError("current Survival model feature contract mismatch")
            features = load_current_monitoring_feature_rows(root, manifest)
            expected_rows = len(features) * len(HORIZON_GRID)
            complete = coverage_complete(features, _coverage(cursor, manifest["state_id"], selection))
            if not should_process(_status(cursor, manifest["state_id"], selection), expected_rows=expected_rows, complete=complete):
                connection.commit()
                return {"status": "served", "rows": 0}
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection))
            if not features:
                cursor.execute("DELETE FROM analytics.battery_current_survival_predictions WHERE dataset = %s", (selection["dataset"],))
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
            prune_stale_predictions(cursor, features)
            if not coverage_complete(features, _coverage(cursor, manifest["state_id"], selection)):
                raise RuntimeError("current Survival publication is incomplete")
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection, status="served", rows_written=expected_rows))
            connection.commit()
            return {"status": "served", "rows": expected_rows}
        except Exception as error:
            upsert_serving_status(cursor, serving_status_row("MATR", manifest["state_id"], "survival_current", selection,
                status="failed", error_message=str(error)[:500]))
            connection.commit()
            return {"status": "failed", "rows": 0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/matr"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Refresh the current finalized state once, then exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    import psycopg
    while True:
        try:
            with psycopg.connect(os.environ["DATABASE_URL"], row_factory=psycopg.rows.dict_row) as connection:
                result = process_once(args.root, connection)
                print(result, flush=True)
                if args.once and result["status"] != "served":
                    raise SystemExit(result["status"])
        except Exception as error:
            print(f"survival serving failed: {error}", flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
