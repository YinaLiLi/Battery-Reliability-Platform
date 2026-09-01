"""Read-only Streamlit views over PostgreSQL analytics serving tables."""
import os

import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row

from src.dashboard_data import lifecycle_stage, lowest_rows, model_metrics, soh_percent


@st.cache_resource
def database():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


@st.cache_data(ttl="5m")
def rows(sql, params=None):
    with database().cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()


def evaluations():
    return rows(
        """
        SELECT model_version, model_name, dataset, status, evaluated_at, metrics, training_metadata
        FROM analytics.model_evaluations
        ORDER BY evaluated_at DESC
        """
    )


def fleet_page():
    st.header("Battery reliability monitoring")
    st.caption("Measured SOH is derived from capacity. RUL is ML-predicted by the current champion.")
    fleet = pd.DataFrame(rows("SELECT * FROM analytics.dashboard_battery_latest ORDER BY battery_id"))
    if fleet.empty:
        st.info("No serving data is available.")
        return

    champion = fleet["champion_model_version"].dropna().iloc[0] if fleet["champion_model_version"].notna().any() else None
    metrics = st.columns(3)
    metrics[0].metric("Batteries tracked", len(fleet))
    metrics[1].metric("Average SOH", f"{fleet['measured_soh'].mean():.1%}")
    metrics[2].metric("Median SOH", f"{fleet['measured_soh'].median():.1%}")
    champion_summary = st.columns(2)
    champion_summary[0].markdown("**Current champion model**")
    champion_summary[0].code(champion or "No champion promoted")
    champion_summary[1].markdown("**Batteries with champion RUL available**")
    champion_summary[1].markdown(f"## {int(fleet['predicted_rul_cycles'].notna().sum())}")

    st.subheader("Measured SOH distribution")
    soh_values = fleet["measured_soh"].map(soh_percent)
    histogram = soh_values.groupby(pd.cut(soh_values, bins=10, include_lowest=True), observed=False).size()
    histogram.index = [f"{interval.left:.0f}–{interval.right:.0f}%" for interval in histogram.index]
    st.bar_chart(histogram, x_label="Measured SOH (%)", y_label="Batteries")
    st.caption("No high-risk label is applied; reliability thresholds have not been approved.")

    rankings = st.columns(2)
    lowest_soh = pd.DataFrame(lowest_rows(fleet.to_dict("records"), "measured_soh"))
    lowest_soh["Measured SOH (%)"] = lowest_soh["measured_soh"].map(soh_percent)
    rankings[0].subheader("Lowest measured SOH")
    rankings[0].dataframe(lowest_soh[["battery_id", "current_cycle", "Measured SOH (%)"]].rename(columns={"battery_id": "Battery", "current_cycle": "Current cycle"}), hide_index=True, width="stretch", column_config={"Measured SOH (%)": st.column_config.NumberColumn(format="%.1f")})
    rankings[1].subheader("Lowest champion-predicted RUL")
    lowest_rul = pd.DataFrame(lowest_rows(fleet.to_dict("records"), "predicted_rul_cycles"))
    if lowest_rul.empty:
        rankings[1].caption("No champion prediction is available.")
    else:
        rankings[1].dataframe(lowest_rul[["battery_id", "current_cycle", "predicted_rul_cycles", "estimated_eol_cycle"]].rename(columns={"battery_id": "Battery", "current_cycle": "Current cycle", "predicted_rul_cycles": "Predicted RUL (cycles)", "estimated_eol_cycle": "Estimated EOL cycle"}), hide_index=True, width="stretch")

    st.subheader("Battery table")
    with st.expander("Filters", expanded=False):
        filters = st.columns(3)
        search = filters[0].text_input("Battery ID")
        min_soh = filters[1].number_input("Minimum SOH (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        maximum_available_rul = fleet["predicted_rul_cycles"].max()
        max_rul = filters[2].number_input("Maximum predicted RUL (cycles)", min_value=0.0, value=0.0 if pd.isna(maximum_available_rul) else float(maximum_available_rul), step=10.0, disabled=not bool(champion))
    filtered = fleet[fleet["battery_id"].str.contains(search, case=False, na=False)]
    filtered = filtered[filtered["measured_soh"].map(soh_percent) >= min_soh]
    if champion:
        filtered = filtered[filtered["predicted_rul_cycles"].fillna(float("inf")) <= max_rul]
    filtered = filtered.assign(lifecycle_stage=[lifecycle_stage(row.current_cycle, row.predicted_rul_cycles) for row in filtered.itertuples()])
    filtered["measured_soh_percent"] = filtered["measured_soh"].map(soh_percent)
    visible = filtered.rename(
        columns={
            "battery_id": "Battery",
            "current_cycle": "Current cycle",
            "measured_soh_percent": "Measured SOH (%)",
            "predicted_rul_cycles": "Champion predicted RUL (cycles)",
            "estimated_eol_cycle": "Estimated EOL cycle",
            "lifecycle_stage": "Lifecycle stage",
            "prediction_created_at": "Prediction timestamp",
        }
    )
    event = st.dataframe(
        visible[["Battery", "Current cycle", "Measured SOH (%)", "Champion predicted RUL (cycles)", "Estimated EOL cycle", "Lifecycle stage", "Prediction timestamp"]],
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        column_config={"Measured SOH (%)": st.column_config.NumberColumn(format="%.1f")},
    )
    if event.selection.rows:
        st.session_state.battery_id = filtered.iloc[event.selection.rows[0]].battery_id
        st.session_state.page = "Battery detail"
        st.rerun()


def battery_page():
    st.header("Battery detail")
    fleet = pd.DataFrame(rows("SELECT * FROM analytics.dashboard_battery_latest ORDER BY battery_id"))
    if fleet.empty:
        st.info("No serving data is available.")
        return
    battery_id = st.selectbox("Battery", fleet["battery_id"], index=list(fleet["battery_id"]).index(st.session_state.get("battery_id", fleet.iloc[0].battery_id)))
    st.session_state.battery_id = battery_id
    latest = fleet.loc[fleet["battery_id"] == battery_id].iloc[0]
    cards = st.columns(5)
    cards[0].metric("Current cycle", int(latest.current_cycle))
    cards[1].metric("Measured SOH", f"{latest.measured_soh:.1%}")
    cards[2].metric("Predicted RUL", "Unavailable" if pd.isna(latest.predicted_rul_cycles) else f"{latest.predicted_rul_cycles:.0f} cycles")
    cards[3].metric("Estimated EOL cycle", "Unavailable" if pd.isna(latest.estimated_eol_cycle) else f"{latest.estimated_eol_cycle:.0f}")
    cards[4].metric("Lifecycle stage", lifecycle_stage(latest.current_cycle, latest.predicted_rul_cycles))
    st.caption("Measured SOH/capacity is not an ML prediction. Candidate RUL is never presented as the fleet’s operational forecast.")

    health = pd.DataFrame(rows(
        """
        SELECT cycle_index, soh AS measured_soh, discharge_capacity_in_ah AS measured_capacity_in_ah,
               temperature_max_in_c, internal_resistance_in_ohm
        FROM analytics.battery_cycle_health
        WHERE dataset = %(dataset)s AND battery_id = %(battery_id)s
        ORDER BY cycle_index
        """,
        {"dataset": latest.dataset, "battery_id": battery_id},
    ))
    st.subheader("Measured degradation")
    degradation = st.columns(2)
    health["measured_soh_percent"] = health["measured_soh"].map(soh_percent)
    degradation[0].caption("Measured SOH (%)")
    degradation[0].line_chart(health.set_index("cycle_index")[["measured_soh_percent"]])
    degradation[1].caption("Measured discharge capacity (Ah)")
    degradation[1].line_chart(health.set_index("cycle_index")[["measured_capacity_in_ah"]])
    trends = st.columns(2)
    trends[0].caption("Maximum temperature (°C)")
    trends[0].line_chart(health.set_index("cycle_index")[["temperature_max_in_c"]])
    trends[1].caption("Internal resistance (Ω)")
    trends[1].line_chart(health.set_index("cycle_index")[["internal_resistance_in_ohm"]])

    models = evaluations()
    versions = [model["model_version"] for model in models]
    champion = next((model["model_version"] for model in models if model["status"] == "champion"), None)
    if not versions:
        st.info("No model prediction history is available.")
        return
    selected = st.selectbox("RUL model history", versions, index=versions.index(champion) if champion else 0)
    selected_status = next(model["status"] for model in models if model["model_version"] == selected)
    predictions = pd.DataFrame(rows(
        """
        SELECT cycle_index, predicted_rul_cycles, cycle_index + predicted_rul_cycles AS estimated_eol_cycle
        FROM analytics.battery_predictions
        WHERE model_version = %(model_version)s AND dataset = %(dataset)s AND battery_id = %(battery_id)s
        ORDER BY cycle_index
        """,
        {"model_version": selected, "dataset": latest.dataset, "battery_id": battery_id},
    ))
    st.subheader("ML-predicted RUL history")
    st.caption(f"Selected model status: {selected_status}. {'Operational champion forecast.' if selected_status == 'champion' else 'Candidate result for evaluation only.'}")
    if predictions.empty:
        st.info("This model has no prediction history for the selected battery.")
    else:
        st.line_chart(predictions.set_index("cycle_index")[["predicted_rul_cycles", "estimated_eol_cycle"]])


def model_page():
    st.header("Model monitoring")
    models = evaluations()
    if not models:
        st.info("No model evaluations are available.")
        return
    champion = next((model for model in models if model["status"] == "champion"), None)
    st.metric("Current champion", champion["model_version"] if champion else "No champion promoted")
    st.caption("This view is read-only. Candidate promotion remains a future human-controlled database operation.")
    flattened = pd.DataFrame([model_metrics(model) for model in models])
    st.dataframe(flattened, hide_index=True, width="stretch")
    candidates = [model for model in models if model["status"] == "candidate"]
    if champion and candidates:
        candidate_version = st.selectbox("Compare champion with candidate", [model["model_version"] for model in candidates])
        candidate = next(model for model in candidates if model["model_version"] == candidate_version)
        comparison = pd.DataFrame([model_metrics(champion), model_metrics(candidate)]).set_index("Model version")
        st.bar_chart(comparison[["Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"]])
    elif candidates:
        st.info("Candidates are available, but none has been promoted to champion for comparison.")


st.set_page_config(page_title="MATR Battery Reliability", layout="wide")
st.sidebar.title("MATR reliability")
if "page" not in st.session_state:
    st.session_state.page = "Fleet monitoring"
st.sidebar.radio("View", ["Fleet monitoring", "Battery detail", "Model monitoring"], key="page")
{"Fleet monitoring": fleet_page, "Battery detail": battery_page, "Model monitoring": model_page}[st.session_state.page]()
