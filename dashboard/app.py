"""Read-only Streamlit views over PostgreSQL analytics serving tables."""
import os

import altair as alt
import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row

from src.dashboard_data import lifecycle_stage, model_display_names, model_metrics, soh_percent


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
        SELECT model_version, model_name, dataset, status, evaluated_at, metrics, training_metadata, model_fingerprint, generation
        FROM analytics.model_evaluations
        WHERE status IN ('candidate', 'champion')
        ORDER BY evaluated_at DESC
        """
    )


def fleet_page():
    st.header("Battery reliability monitoring")
    fleet = pd.DataFrame(rows("SELECT * FROM analytics.dashboard_battery_latest ORDER BY battery_id"))
    if fleet.empty:
        st.info("No serving data is available.")
        return

    champion = fleet["champion_model_version"].dropna().iloc[0] if fleet["champion_model_version"].notna().any() else None
    model_names = model_display_names(evaluations())
    champion_name = model_names.get(champion, "No champion promoted")
    st.caption(f"Measured SOH is derived from capacity. RUL is predicted by {champion_name}.")
    metrics = st.columns(5)
    metrics[0].metric("Batteries tracked", len(fleet))
    metrics[1].metric("Average SOH", f"{fleet['measured_soh'].mean():.1%}")
    metrics[2].metric("Median SOH", f"{fleet['measured_soh'].median():.1%}")
    metrics[3].metric("Current champion model", champion_name)
    metrics[4].metric("RUL predictions available", int(fleet["predicted_rul_cycles"].notna().sum()))

    st.subheader("Measured SOH distribution")
    soh_values = fleet["measured_soh"].map(soh_percent)
    histogram = soh_values.groupby(pd.cut(soh_values, bins=10, include_lowest=True), observed=False).size()
    histogram.index = [f"{interval.left:.0f}–{interval.right:.0f}%" for interval in histogram.index]
    st.bar_chart(histogram, x_label="Measured SOH (%)", y_label="Batteries")
    st.caption(f"Measured SOH is derived from capacity. RUL is predicted by {champion_name}.")

    st.subheader("Battery table")
    with st.expander("Filters", expanded=False):
        filters = st.columns(3)
        search = filters[0].text_input("Battery ID")
        measured_soh = fleet["measured_soh"].map(soh_percent)
        min_soh = filters[1].number_input(
            "Minimum SOH (%)",
            min_value=float(measured_soh.min()),
            max_value=float(measured_soh.max()),
            value=float(measured_soh.min()),
            step=1.0,
        )
        available_rul = fleet["predicted_rul_cycles"].dropna()
        available_rul_max = 0.0 if available_rul.empty else max(0.0, float(available_rul.max()))
        max_rul = filters[2].number_input(
            "Maximum predicted RUL (cycles)",
            min_value=0.0,
            max_value=available_rul_max,
            value=available_rul_max,
            step=1.0,
        )
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
        st.session_state.navigate_to = "Battery detail"
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
    models = evaluations()
    model_names = model_display_names(models)
    champion_name = model_names.get(latest.champion_model_version, "No champion promoted")
    cards = st.columns(5)
    cards[0].metric("Current cycle", int(latest.current_cycle))
    cards[1].metric("Measured SOH", f"{latest.measured_soh:.1%}")
    cards[2].metric(f"Predicted RUL · {champion_name}", "Unavailable" if pd.isna(latest.predicted_rul_cycles) else f"{latest.predicted_rul_cycles:.0f} cycles")
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

    versions = [model["model_version"] for model in models]
    champion = next((model["model_version"] for model in models if model["status"] == "champion"), None)
    if not versions:
        st.info("No model prediction history is available.")
        return
    selected = st.selectbox("RUL model history", versions, index=versions.index(champion) if champion else 0, format_func=lambda version: model_names.get(version, version))
    selected_status = next(model["status"] for model in models if model["model_version"] == selected)
    predictions = pd.DataFrame(rows(
        """
        SELECT cycle_index, raw_predicted_rul_cycles, predicted_rul_cycles, predicted_eol_cycle
        FROM analytics.battery_predictions
        WHERE model_version = %(model_version)s AND dataset = %(dataset)s AND battery_id = %(battery_id)s
        ORDER BY cycle_index
        """,
        {"model_version": selected, "dataset": latest.dataset, "battery_id": battery_id},
    ))
    st.subheader("ML-predicted RUL history")
    st.caption(f"{model_names.get(selected, selected)} · internal version: {selected}. Selected model status: {selected_status}. {'Operational champion forecast.' if selected_status == 'champion' else 'Candidate result for evaluation only.'}")
    if predictions.empty:
        st.info("This model has no prediction history for the selected battery.")
    else:
        first_eol = predictions.loc[predictions["predicted_rul_cycles"] == 0, "cycle_index"].min()
        predictions["estimated_eol_cycle"] = predictions["predicted_eol_cycle"].where(predictions["cycle_index"] <= first_eol) if pd.notna(first_eol) else predictions["predicted_eol_cycle"]
        history = predictions.melt("cycle_index", ["predicted_rul_cycles", "estimated_eol_cycle"], var_name="series", value_name="cycles").dropna()
        chart = alt.Chart(history).mark_line().encode(x="cycle_index:Q", y="cycles:Q", color="series:N")
        if pd.notna(first_eol):
            marker = predictions.loc[predictions["cycle_index"] == first_eol]
            chart += alt.Chart(marker).mark_point(color="#d62728", filled=True, size=90).encode(x="cycle_index:Q", y="predicted_eol_cycle:Q")
        st.altair_chart(chart, width="stretch")
        if pd.notna(first_eol):
            st.caption(f"Predicted EOL reached at cycle {int(first_eol)}; the served EOL is frozen there.")
        with st.expander("Raw model diagnostics"):
            st.caption("Raw model output is retained for diagnostics only; charts above use operational served predictions.")
            st.dataframe(predictions[["cycle_index", "raw_predicted_rul_cycles"]], hide_index=True, width="stretch")


def model_page():
    st.header("Model monitoring")
    models = evaluations()
    if not models:
        st.info("No model evaluations are available.")
        return
    champion = next((model for model in models if model["status"] == "champion"), None)
    model_names = model_display_names(models)
    champion_name = model_names.get(champion["model_version"], "No champion promoted") if champion else "No champion promoted"
    st.metric("Current champion", champion_name)
    st.caption("This view is read-only. Candidate promotion remains a future human-controlled database operation.")
    flattened = pd.DataFrame([model_metrics(model) for model in models])
    flattened.insert(0, "Display model", [model_names.get(model["model_version"], model["model_version"]) for model in models])
    flattened = flattened.rename(columns={"Model version": "Internal model version"})
    st.dataframe(flattened, hide_index=True, width="stretch")

    st.subheader("Metric definitions")
    st.markdown(
        """- Test MAE — On average, how many cycles the prediction is off by; lower is better.
- Test RMSE — Similar to MAE, but gives more weight to large prediction errors; lower is better.
- Lifecycle MAE — Shows how prediction error changes from early to mid to late battery life; lower is better.
- R² — How well the model explains differences in remaining battery life; closer to 1 is better."""
    )

    candidates = [model for model in models if model["status"] == "candidate"]
    baseline = champion or min(candidates, key=lambda model: model_names.get(model["model_version"], model["model_version"]), default=None)
    if not baseline:
        st.info("No candidate or champion model is available for comparison.")
        return
    if not champion:
        st.caption("No champion is promoted; comparison uses the first candidate as its evaluation baseline.")
    candidate_options = [model["model_version"] for model in candidates if model["model_version"] != baseline["model_version"]]
    selected_candidates = st.multiselect(
        "Add models to compare",
        options=candidate_options,
        default=[],
        format_func=lambda version: model_names.get(version, version),
        max_selections=4,
        help="The champion, or the first candidate when no champion is promoted, is always included.",
    )
    selected_versions = [baseline["model_version"], *selected_candidates]
    selected_models = [model for model in models if model["model_version"] in selected_versions]
    if not selected_models:
        return

    comparison = pd.DataFrame([model_metrics(model) for model in selected_models])
    comparison["Display model"] = [model_names.get(model["model_version"], model["model_version"]) for model in selected_models]
    if len(selected_versions) > 1:
        errors = comparison.melt(
            id_vars=["Display model"],
            value_vars=["Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"],
            var_name="Metric",
            value_name="Cycles of error",
        )
        st.subheader("Error metric comparison")
        st.caption("Lower is better")
        st.altair_chart(
            alt.Chart(errors).mark_bar().encode(
                x=alt.X("Metric:N", sort=["Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"]),
                y="Cycles of error:Q",
                xOffset="Display model:N",
                color="Display model:N",
            ),
            width="stretch",
        )

    st.subheader("R² comparison")
    st.caption("Higher is better")
    r2 = comparison[["Display model", "Test R²"]].rename(columns={"Test R²": "R²"})
    st.dataframe(r2, hide_index=True, width="stretch")

    st.subheader("Model metadata")
    for model in selected_models:
        display_name = model_names.get(model["model_version"], model["model_version"])
        heading = f"{display_name} ({'champion' if model['status'] == 'champion' else model['status']})"
        with st.expander(heading):
            st.markdown(f"**Internal model version:** `{model['model_version']}`")
            st.markdown(f"**Status:** {model['status']}")
            st.markdown(f"**Evaluation timestamp:** {model['evaluated_at']}")
            st.markdown(f"**Validation MAE:** {model_metrics(model)['Validation MAE']:.4f}")
            st.markdown(f"**Test MAE:** {model_metrics(model)['Test MAE']:.4f}")
            st.markdown(f"**RMSE:** {model_metrics(model)['Test RMSE']:.4f}")
            st.markdown(f"**R²:** {model_metrics(model)['Test R²']:.4f}")
            st.markdown(f"**Early lifecycle MAE:** {model_metrics(model)['Early MAE']:.4f}")
            st.markdown(f"**Mid lifecycle MAE:** {model_metrics(model)['Mid MAE']:.4f}")
            st.markdown(f"**Late lifecycle MAE:** {model_metrics(model)['Late MAE']:.4f}")
            if model.get("training_metadata"):
                st.markdown("**Training metadata:**")
                st.json(model["training_metadata"])


st.set_page_config(page_title="MATR Battery Reliability", layout="wide")
st.sidebar.title("MATR reliability")
if "navigate_to" in st.session_state:
    st.session_state.page = st.session_state.pop("navigate_to")
if "page" not in st.session_state:
    st.session_state.page = "Fleet monitoring"
st.sidebar.radio("View", ["Fleet monitoring", "Battery detail", "Model monitoring"], key="page")
{"Fleet monitoring": fleet_page, "Battery detail": battery_page, "Model monitoring": model_page}[st.session_state.page]()
