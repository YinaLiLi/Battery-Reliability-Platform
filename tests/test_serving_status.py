from src.serving_status import current_stream_state_row, serving_status_row


def test_current_stream_state_is_state_only_and_published_before_serving():
    manifest = {"state_id": "state-1", "feature_contract_version": "features-v1"}

    assert current_stream_state_row("MATR", manifest, "2026-01-01T00:00:00+00:00") == {
        "dataset": "MATR", "state_id": "state-1", "feature_contract_version": "features-v1",
        "published_at": "2026-01-01T00:00:00+00:00",
    }


def test_missing_current_model_is_explicitly_unavailable():
    assert serving_status_row("MATR", "state-1", "survival_current", None)["status"] == "unavailable"
    assert serving_status_row("MATR", "state-1", "survival_current", None)["selection_revision"] == 0


def test_selected_model_status_keeps_its_selection_revision():
    row = serving_status_row("MATR", "state-1", "rul_current", {
        "model_version": "model-1.2", "model_fingerprint": "fingerprint", "selection_revision": 1,
    }, status="served")

    assert row["selection_revision"] == 1
    assert row["model_version"] == "model-1.2"
