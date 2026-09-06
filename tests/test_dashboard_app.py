import ast
from pathlib import Path


APP = Path(__file__).parents[1] / "dashboard" / "app.py"
ANALYTICS_SQL = Path(__file__).parents[1] / "sql" / "001_analytics.sql"


def function_source(name):
    source = APP.read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source, function)


def test_dashboard_view_joins_full_prediction_history_to_the_current_model():
    fleet_page = function_source("fleet_page")
    battery_page = function_source("battery_page")
    sql = ANALYTICS_SQL.read_text()

    assert "LEFT JOIN analytics.current_models AS current" in sql
    assert "prediction.model_version = current.model_version" in sql
    assert "), champion AS (" not in sql
    assert "battery_current_predictions" not in fleet_page
    assert "battery_current_predictions" not in battery_page


def test_survival_page_uses_one_selected_artifact_for_table_and_winner_metrics():
    survival_page = function_source("survival_model_page")

    assert 'selected = next(model for model in models if model["model_version"] == selected_version)' in survival_page
    assert 'model_display_names(models)' in survival_page
    assert '"Internal model version"' in survival_page
