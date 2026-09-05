import pytest

from src.postgres_loader import build_current_prediction_upsert_sql, build_current_survival_prediction_upsert_sql, build_current_stream_state_upsert_sql
from src.spark_streaming import CURRENT_PREDICTION_SCHEMA


def test_current_prediction_upsert_only_replaces_an_older_stream_position():
    sql = build_current_prediction_upsert_sql("analytics.current_prediction_stage")

    assert "INSERT INTO analytics.battery_current_predictions" in sql
    assert "ON CONFLICT (dataset, battery_id) DO UPDATE" in sql
    assert "EXCLUDED.replay_sequence > analytics.battery_current_predictions.replay_sequence" in sql
    assert "EXCLUDED.state_id = analytics.battery_current_predictions.state_id" in sql
    assert "EXCLUDED.selection_revision > analytics.battery_current_predictions.selection_revision" in sql
    assert "analytics.current_prediction_stage" in sql


def test_current_prediction_upsert_rejects_an_unsafe_staging_table_name():
    with pytest.raises(ValueError, match="safe PostgreSQL identifier"):
        build_current_prediction_upsert_sql("analytics.stage; DROP TABLE analytics.current_models")


def test_current_prediction_staging_keeps_timestamp_castable():
    assert CURRENT_PREDICTION_SCHEMA["inference_created_at"].dataType.simpleString() == "string"


def test_current_survival_upsert_only_replaces_an_older_stream_position():
    sql = build_current_survival_prediction_upsert_sql("analytics.current_survival_stage")

    assert "INSERT INTO analytics.battery_current_survival_predictions" in sql
    assert "ON CONFLICT (dataset, battery_id, horizon_cycles) DO UPDATE" in sql
    assert "EXCLUDED.replay_sequence > analytics.battery_current_survival_predictions.replay_sequence" in sql
    assert "EXCLUDED.selection_revision > analytics.battery_current_survival_predictions.selection_revision" in sql


def test_current_stream_state_upsert_exposes_only_the_latest_finalized_state():
    sql = build_current_stream_state_upsert_sql("analytics.stream_state_stage")

    assert "INSERT INTO analytics.current_stream_states" in sql
    assert "ON CONFLICT (dataset) DO UPDATE" in sql
    assert "analytics.stream_state_stage" in sql
