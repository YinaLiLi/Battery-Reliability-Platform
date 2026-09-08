"""Read-only Streamlit views over PostgreSQL analytics serving tables."""
import os

import altair as alt
import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row

from src.dashboard_data import current_survival_curve, family_validation_rows, filter_batteries_by_risk, fleet_descriptive_kpis, lifecycle_stage, latest_model_version, measured_soh_distribution, model_display_names, model_metrics, model_selector_names, performance_gradient, selectable_models, serving_models, soh_percent, survival_family_validation_rows, survival_model_metrics


@st.cache_resource
def database():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


@st.cache_data(ttl="5m")
def rows(sql, params=None):
    for attempt in range(2):
        try:
            with database().cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except (psycopg.errors.AdminShutdown, psycopg.OperationalError):
            if attempt:
                raise
            database.clear()


@st.cache_data(ttl="30s")
def serving_rows(sql, params=None):
    for attempt in range(2):
        try:
            with database().cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except (psycopg.errors.AdminShutdown, psycopg.OperationalError):
            if attempt:
                raise
            database.clear()


def persist_current_model(dataset):
    """Make the manually selected model visible to the shared inference worker."""
    model_version = st.session_state.get("current_model_version")
    if not model_version:
        return
    with database().cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO analytics.current_models (dataset, model_version, model_fingerprint, selection_revision, updated_at)
            SELECT %(dataset)s, model_version, model_fingerprint, 1, NOW()
            FROM analytics.model_evaluations
            WHERE dataset = %(dataset)s AND model_version = %(model_version)s
              AND status IN ('candidate', 'champion')
            ON CONFLICT (dataset) DO UPDATE
            SET model_version = EXCLUDED.model_version,
                model_fingerprint = EXCLUDED.model_fingerprint,
                selection_revision = analytics.current_models.selection_revision + 1,
                updated_at = NOW()
            WHERE analytics.current_models.model_version IS DISTINCT FROM EXCLUDED.model_version
               OR analytics.current_models.model_fingerprint IS DISTINCT FROM EXCLUDED.model_fingerprint
            """,
            {"dataset": dataset, "model_version": model_version},
        )
    database().commit()
    serving_rows.clear()
    rows.clear()


def evaluations():
    return rows(
        """
        SELECT model_version, model_name, dataset, status, evaluated_at, metrics, training_metadata, model_fingerprint, generation
        FROM analytics.model_evaluations
        WHERE status IN ('candidate', 'champion')
        ORDER BY evaluated_at DESC
        """
    )


def current_model(models):
    if not models:
        return None
    persisted = serving_rows("SELECT model_version FROM analytics.current_models WHERE dataset = %(dataset)s", {"dataset": models[0]["dataset"]})
    persisted_version = persisted[0]["model_version"] if persisted else None
    models = serving_models(models)
    versions = [model["model_version"] for model in models]
    selected = persisted_version or st.session_state.get("current_model_version")
    if selected not in versions:
        selected = latest_model_version(models)
    if st.session_state.get("current_model_version") != selected:
        st.session_state.current_model_version = selected
    return next(model for model in models if model["model_version"] == selected)


def survival_evaluations():
    return rows("SELECT model_version, model_name, dataset, status, evaluated_at, metrics, training_metadata, model_fingerprint, generation FROM analytics.survival_model_evaluations WHERE status IN ('candidate', 'champion') ORDER BY evaluated_at DESC")


def persist_current_survival_model(dataset):
    """Persist the independent manual Survival selection for the stream worker."""
    model_version = st.session_state.get("current_survival_model_version")
    if not model_version:
        return
    with database().cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO analytics.current_survival_models (dataset, model_version, model_fingerprint, selection_revision, updated_at)
            SELECT %(dataset)s, model_version, model_fingerprint, 1, NOW()
            FROM analytics.survival_model_evaluations
            WHERE dataset = %(dataset)s AND model_version = %(model_version)s
              AND status IN ('candidate', 'champion')
            ON CONFLICT (dataset) DO UPDATE
            SET model_version = EXCLUDED.model_version,
                model_fingerprint = EXCLUDED.model_fingerprint,
                selection_revision = analytics.current_survival_models.selection_revision + 1,
                updated_at = NOW()
            WHERE analytics.current_survival_models.model_version IS DISTINCT FROM EXCLUDED.model_version
               OR analytics.current_survival_models.model_fingerprint IS DISTINCT FROM EXCLUDED.model_fingerprint
            """,
            {"dataset": dataset, "model_version": model_version},
        )
    database().commit()
    serving_rows.clear()
    rows.clear()


def current_survival_model(models):
    if not models:
        return None
    persisted = serving_rows("SELECT model_version FROM analytics.current_survival_models WHERE dataset = %(dataset)s", {"dataset": models[0]["dataset"]})
    persisted_version = persisted[0]["model_version"] if persisted else None
    models = serving_models(models)
    versions = [model["model_version"] for model in models]
    selected = persisted_version or st.session_state.get("current_survival_model_version", versions[0])
    if selected not in versions:
        selected = versions[0]
    if st.session_state.get("current_survival_model_version") != selected:
        st.session_state.current_survival_model_version = selected
    return next(model for model in models if model["model_version"] == selected)


def current_survival_serving_state(dataset):
    return serving_rows("""
        SELECT state.state_id, status.status, status.model_version, status.model_fingerprint,
               status.selection_revision, status.error_message
        FROM analytics.current_stream_states AS state
        LEFT JOIN analytics.current_survival_models AS current USING (dataset)
        LEFT JOIN analytics.stream_serving_status AS status
          ON status.dataset = state.dataset AND status.state_id = state.state_id
         AND status.consumer = 'survival_current'
         AND status.selection_revision = COALESCE(current.selection_revision, 0)
         AND status.model_version IS NOT DISTINCT FROM current.model_version
         AND status.model_fingerprint IS NOT DISTINCT FROM current.model_fingerprint
        WHERE state.dataset = %(dataset)s
    """, {"dataset": dataset})


def fleet_page():
    st.header("Battery reliability monitoring")
    fleet = pd.DataFrame(rows("SELECT * FROM analytics.dashboard_battery_latest ORDER BY battery_id"))
    if fleet.empty:
        st.info("No serving data is available.")
        return

    models = evaluations()
    selected_model = current_model(models)
    selected_version = selected_model["model_version"] if selected_model else None
    selected_name = model_display_names(models).get(selected_version, "Unavailable")
    survival_models = survival_evaluations()
    selected_survival = current_survival_model(survival_models)
    selected_survival_version = selected_survival["model_version"] if selected_survival else None
    survival_values = pd.DataFrame(rows(
        """
        SELECT state.battery_id, prediction.survival_probability AS survival_probability_100_cycles
        FROM analytics.dashboard_battery_latest AS state
        JOIN analytics.current_survival_models AS current USING (dataset)
        JOIN analytics.battery_current_survival_predictions AS prediction
          ON prediction.dataset = state.dataset
         AND prediction.battery_id = state.battery_id
         AND prediction.cycle_index = state.current_cycle
         AND prediction.model_version = current.model_version
         AND prediction.model_fingerprint = current.model_fingerprint
         AND prediction.selection_revision = current.selection_revision
         AND prediction.horizon_cycles = 100
        WHERE state.dataset = %(dataset)s
        """,
        {"dataset": fleet.iloc[0]["dataset"]},
    ))
    if survival_values.empty:
        fleet["survival_probability_100_cycles"] = pd.NA
    else:
        fleet = fleet.merge(survival_values, on="battery_id", how="left")

    st.subheader("Current model")
    active = serving_models(models)
    versions = [model["model_version"] for model in active]
    selector_names = model_selector_names(active)
    if versions:
        st.selectbox(
            "Current model",
            versions,
            index=versions.index(selected_version) if selected_version in versions else 0,
            key="current_model_version",
            on_change=persist_current_model,
            args=(fleet.iloc[0]["dataset"],),
            format_func=lambda version: selector_names.get(version, version),
            label_visibility="collapsed",
        )
    selected_model = current_model(models)
    selected_version = selected_model["model_version"] if selected_model else None
    selected_name = model_display_names(models).get(selected_version, "Unavailable")

    rul_predictions_available = int(fleet["predicted_rul_cycles"].notna().sum())
    descriptive_kpis = fleet_descriptive_kpis(fleet)

    def cycles(value):
        return "Unavailable" if value is None else f"{value:.0f} cycles"

    def percentage(value):
        return "Unavailable" if value is None else f"{value:.1f}%"

    metric_row = st.columns(4)
    metric_row[0].metric("Batteries tracked", len(fleet))
    metric_row[1].metric("Average SOH", f"{fleet['measured_soh'].mean():.1%}")
    metric_row[2].metric("Median SOH", f"{fleet['measured_soh'].median():.1%}")
    metric_row[3].metric("RUL predictions available", rul_predictions_available)
    serving_kpi_row = st.columns(4)
    serving_kpi_row[0].metric("Median predicted RUL", cycles(descriptive_kpis["median_predicted_rul_cycles"]))
    serving_kpi_row[1].metric("P25 predicted RUL", cycles(descriptive_kpis["p25_predicted_rul_cycles"]))
    serving_kpi_row[2].metric("Median +100 survival", percentage(descriptive_kpis["median_survival_probability_100_pct"]))
    serving_kpi_row[3].metric("P25 +100 survival", percentage(descriptive_kpis["p25_survival_probability_100_pct"]))
    st.caption(f"Measured SOH is derived from capacity. RUL is predicted by {selected_name}; +100-cycle survival is from {model_display_names(survival_models).get(selected_survival_version, 'Unavailable')}.")

    st.subheader("Measured SOH distribution")
    histogram = measured_soh_distribution(fleet)
    st.altair_chart(
        alt.Chart(histogram).mark_bar().encode(
            x=alt.X("Measured SOH bin:N", sort=None, title="Measured SOH (%)"),
            y=alt.Y("Battery count:Q", title="Batteries (count)", axis=alt.Axis(format="d")),
            tooltip=["Measured SOH bin:N", "Battery count:Q"],
        ),
        width="stretch",
    )

    st.subheader("Battery table")
    with st.expander("Filters", expanded=False):
        filters = st.columns(4)
        search = filters[0].text_input("Battery ID")
        measured_soh = fleet["measured_soh"].map(soh_percent)
        max_soh = filters[1].number_input(
            "Maximum SOH (%)",
            min_value=float(measured_soh.min()),
            max_value=float(measured_soh.max()),
            value=float(measured_soh.max()),
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
        max_survival = filters[3].number_input(
            "Maximum +100-cycle survival probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=1.0,
        )
    filtered = filter_batteries_by_risk(
        fleet,
        search,
        max_soh=max_soh,
        max_rul=max_rul if selected_version else float("inf"),
        max_survival=max_survival if selected_survival_version else 100.0,
    )
    filtered = filtered.assign(lifecycle_stage=[lifecycle_stage(row.current_cycle, row.predicted_rul_cycles) for row in filtered.itertuples()])
    filtered["measured_soh_percent"] = filtered["measured_soh"].map(soh_percent)
    filtered["survival_probability_100_cycles_percent"] = filtered["survival_probability_100_cycles"].map(soh_percent)
    visible = filtered.rename(
        columns={
            "battery_id": "Battery",
            "current_cycle": "Current cycle",
            "measured_soh_percent": "Measured SOH (%)",
            "predicted_rul_cycles": "Predicted RUL (cycles)",
            "survival_probability_100_cycles_percent": "Survive +100 cycles (%)",
            "estimated_eol_cycle": "Estimated EOL cycle",
            "lifecycle_stage": "Lifecycle stage",
            "prediction_created_at": "Prediction timestamp",
        }
    )
    visible["Predicted RUL (cycles)"] = visible["Predicted RUL (cycles)"].round()
    visible["Estimated EOL cycle"] = visible["Estimated EOL cycle"].round()
    event = st.dataframe(
        visible[["Battery", "Current cycle", "Measured SOH (%)", "Predicted RUL (cycles)", "Survive +100 cycles (%)", "Estimated EOL cycle", "Lifecycle stage", "Prediction timestamp"]],
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        column_config={
            "Measured SOH (%)": st.column_config.NumberColumn(format="%.1f"),
            "Predicted RUL (cycles)": st.column_config.NumberColumn(format="%.0f"),
            "Survive +100 cycles (%)": st.column_config.NumberColumn(format="%.1f"),
            "Estimated EOL cycle": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    if event.selection.rows:
        st.session_state.battery_id = filtered.iloc[event.selection.rows[0]].battery_id
        st.session_state.navigate_to = "Battery Detail"
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
    selected_model = current_model(models)
    selected_version = selected_model["model_version"] if selected_model else None
    selected_name = model_names.get(selected_version, "Unavailable")
    latest_rul = latest.predicted_rul_cycles if selected_version else None
    latest_eol = latest.predicted_eol_cycle if selected_version else None
    cards = st.columns(5)
    cards[0].metric("Current cycle", int(latest.current_cycle))
    cards[1].metric("Measured SOH", f"{latest.measured_soh:.1%}")
    cards[2].metric(f"Predicted RUL · {selected_name}", "Unavailable" if latest_rul is None else f"{latest_rul:.0f} cycles")
    cards[3].metric("Estimated EOL cycle", "Unavailable" if latest_eol is None else f"{latest_eol:.0f}")
    cards[4].metric("Lifecycle stage", lifecycle_stage(latest.current_cycle, latest_rul))
    st.caption("Measured SOH/capacity is independent of the selected model.")

    survival_models = survival_evaluations()
    survival_names = model_display_names(survival_models)
    selected_survival = current_survival_model(survival_models)
    st.subheader("Survival outlook")
    serving = current_survival_serving_state(latest.dataset)
    if not selected_survival:
        st.info("No persisted Current Survival model is available.")
    elif not serving:
        st.info("Finalized stream state is unavailable.")
    elif serving[0]["status"] != "served":
        status = serving[0]["status"] or "pending"
        detail = serving[0].get("error_message") if status == "failed" else None
        st.info(f"Survival serving is {status}." + (f" {detail}" if detail else ""))
    elif selected_survival:
        curve = pd.DataFrame(rows(
            """SELECT cycle_index, horizon_cycles, survival_probability FROM analytics.battery_current_survival_predictions
               WHERE model_version = %(model_version)s AND dataset = %(dataset)s AND battery_id = %(battery_id)s
                 AND state_id = %(state_id)s AND model_fingerprint = %(model_fingerprint)s
                 AND selection_revision = %(selection_revision)s AND cycle_index = %(cycle_index)s
               ORDER BY horizon_cycles""",
            {"model_version": selected_survival["model_version"], "dataset": latest.dataset, "battery_id": battery_id,
             "state_id": serving[0]["state_id"], "model_fingerprint": serving[0]["model_fingerprint"],
             "selection_revision": serving[0]["selection_revision"], "cycle_index": int(latest.current_cycle)},
        ))
        curve = current_survival_curve(curve, int(latest.current_cycle))
        if not curve.empty:
            st.caption(f"Current Survival model: {survival_names.get(selected_survival['model_version'], selected_survival['model_version'])}. Conditional survival probability is the chance of remaining above the EOL threshold for each horizon, given the current-cycle features.")
            st.altair_chart(
                alt.Chart(curve).mark_line(point=True).encode(
                    x=alt.X("horizon_cycles:Q", title="Horizon (cycles)"),
                    y=alt.Y("survival_probability:Q", title="Conditional survival probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip("horizon_cycles:Q", title="Horizon (cycles)"), alt.Tooltip("survival_probability:Q", title="Conditional survival probability", format=".1%")],
                ),
                width="stretch",
            )
            horizons = curve.set_index("horizon_cycles")["survival_probability"]
            cards = st.columns(3)
            for card, horizon in zip(cards, (50, 100, 200)):
                card.metric(f"Survive +{horizon} cycles", "Unavailable" if horizon not in horizons else f"{horizons[horizon]:.1%}")
        else:
            st.info("No current-cycle Survival prediction is available for this battery.")

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

    if not selected_version:
        st.info("No model prediction history is available.")
        return
    selected = selected_version
    predictions = pd.DataFrame(rows(
        """
        SELECT cycle_index, raw_predicted_rul_cycles, predicted_rul_cycles, predicted_eol_cycle
        FROM analytics.battery_predictions
        WHERE model_version = %(model_version)s AND dataset = %(dataset)s AND battery_id = %(battery_id)s
        ORDER BY cycle_index
        """,
        {"model_version": selected, "dataset": latest.dataset, "battery_id": battery_id},
    ))
    st.subheader("Predicted RUL history")
    st.caption(f"Current model: {model_names.get(selected, selected)}.")
    if predictions.empty:
        st.info("This model has no prediction history for the selected battery.")
    else:
        first_eol = predictions.loc[predictions["predicted_rul_cycles"] == 0, "cycle_index"].min()
        predictions["estimated_eol_cycle"] = predictions["predicted_eol_cycle"].where(predictions["cycle_index"] <= first_eol) if pd.notna(first_eol) else predictions["predicted_eol_cycle"]
        history = predictions.melt("cycle_index", ["predicted_rul_cycles", "estimated_eol_cycle"], var_name="series", value_name="cycles").dropna()
        history["series"] = history["series"].map({"predicted_rul_cycles": "Predicted RUL", "estimated_eol_cycle": "Estimated EOL cycle"})
        chart = alt.Chart(history).mark_line().encode(
            x=alt.X("cycle_index:Q", title="Cycle"),
            y=alt.Y("cycles:Q", title="Cycles"),
            color=alt.Color("series:N", title=None),
        )
        if pd.notna(first_eol):
            marker = predictions.loc[predictions["cycle_index"] == first_eol]
            chart += alt.Chart(marker).mark_point(color="#d62728", filled=True, size=90).encode(x="cycle_index:Q", y="predicted_eol_cycle:Q")
        st.altair_chart(chart, width="stretch")
        if pd.notna(first_eol):
            st.caption(f"Predicted EOL reached at cycle {int(first_eol)}; EOL remains at that cycle thereafter.")
        with st.expander("Raw model diagnostics"):
            st.caption("Raw model output is retained for diagnostics only; charts above use operational served predictions.")
            st.dataframe(predictions[["cycle_index", "raw_predicted_rul_cycles"]], hide_index=True, width="stretch")


def model_page():
    st.header("RUL model monitoring")
    models = evaluations()
    if not models:
        st.info("No model evaluations are available.")
        return
    selected_model = current_model(models)
    active = selectable_models(models, current_version=selected_model["model_version"] if selected_model else None)
    model_names = model_display_names(active)
    if not active:
        st.info("No non-retired canonical model generations are available for monitoring.")
        return
    versions = [model["model_version"] for model in active]
    selector_models = serving_models(models)
    selector_versions = [model["model_version"] for model in selector_models]
    selector_names = model_selector_names(selector_models)
    st.subheader("Current model")
    if selected_model:
        st.selectbox(
            "Current model",
            selector_versions,
            index=selector_versions.index(selected_model["model_version"]),
            key="current_model_version",
            on_change=persist_current_model,
            args=(selected_model["dataset"],),
            format_func=lambda version: selector_names.get(version, version),
            label_visibility="collapsed",
        )
    st.caption("Current is the serving model.")
    flattened = pd.DataFrame([model_metrics(model) for model in active])
    flattened.insert(0, "Display model", [model_names.get(model["model_version"], model["model_version"]) for model in active])
    flattened = flattened.drop(columns=["Model version", "Model fingerprint"], errors="ignore")
    flattened.insert(1, "Selection", ["Current" if model.get("model_version") == st.session_state.get("current_model_version") else "" for model in active])
    st.subheader("Model performance comparison")
    st.dataframe(
        performance_gradient(
            flattened[["Display model", "Selection", "Generation", "Selected model family", "Validation MAE", "Test MAE", "Test RMSE", "Test R²", "Early MAE", "Mid MAE", "Late MAE"]],
            lower_is_better=["Validation MAE", "Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"],
            higher_is_better=["Test R²"],
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Validation MAE": st.column_config.NumberColumn(format="%.1f"),
            "Test MAE": st.column_config.NumberColumn(format="%.1f"),
            "Test RMSE": st.column_config.NumberColumn(format="%.1f"),
            "Test R²": st.column_config.NumberColumn(format="%.3f"),
            "Early MAE": st.column_config.NumberColumn(format="%.1f"),
            "Mid MAE": st.column_config.NumberColumn(format="%.1f"),
            "Late MAE": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    errors = flattened.melt(
        id_vars=["Display model"],
        value_vars=["Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"],
        var_name="Metric",
        value_name="Cycles of error",
    )
    st.altair_chart(
        alt.Chart(errors).mark_bar().encode(
            x=alt.X("Metric:N", sort=["Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"], title="Error metric"),
            y=alt.Y("Cycles of error:Q", title="Cycles"),
            xOffset="Display model:N",
            color=alt.Color("Display model:N", title="Model"),
            tooltip=["Display model", "Metric", "Cycles of error"],
        ),
        width="stretch",
    )
    st.caption("Model selection uses validation metrics only; test metrics are held-out evaluation.")

    st.subheader("Validation comparison")
    generation_version = st.selectbox(
        "Generation",
        [model["model_version"] for model in active],
        format_func=lambda version: model_names.get(version, version),
    )
    generation_model = next(model for model in active if model["model_version"] == generation_version)
    family_validation = pd.DataFrame(family_validation_rows(generation_model))
    if family_validation.empty:
        st.info("This legacy evaluation does not contain family-level validation results.")
    else:
        st.dataframe(
            performance_gradient(
                family_validation[["Model family", "Configuration", "Validation MAE", "Validation RMSE", "Selected"]],
                lower_is_better=["Validation MAE", "Validation RMSE"],
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Validation MAE": st.column_config.NumberColumn(format="%.1f"),
                "Validation RMSE": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.altair_chart(
            alt.Chart(family_validation).mark_bar().encode(
                x=alt.X("Model family:N", sort=["Ridge", "Random Forest", "XGBoost", "MLP"]),
                y="Validation MAE:Q",
                color=alt.Color("Selected:N", scale=alt.Scale(domain=[False, True], range=["#9aa0a6", "#1f77b4"])),
                tooltip=["Model family", "Configuration", "Validation MAE", "Validation RMSE", "Selected"],
            ),
            width="stretch",
        )

    st.caption("MAE: on average, how many cycles the prediction is off by (↓ better)")
    st.caption("RMSE: prediction error that penalizes large misses more heavily (↓ better)")
    st.caption("R²: how much of the variation in actual RUL the model explains (↑ better)")
    st.caption("Lifecycle MAE: prediction error during early, mid, and late battery life (↓ better)")


def survival_model_page():
    st.header("Survival model monitoring")
    all_models = survival_evaluations()
    selected = current_survival_model(all_models)
    models = selectable_models(all_models, current_version=selected["model_version"] if selected else None)
    if not models or not selected:
        st.info("No survival model evaluations are available.")
        return
    model_names = model_display_names(models)
    versions = [model["model_version"] for model in models]
    selector_models = serving_models(all_models)
    selector_versions = [model["model_version"] for model in selector_models]
    selector_names = model_selector_names(selector_models)
    selected_version = selected["model_version"]
    st.subheader("Current model")
    selected_version = st.selectbox("Current survival model", selector_versions, index=selector_versions.index(selected_version), key="current_survival_model_version", on_change=persist_current_survival_model, args=(models[0]["dataset"],), format_func=lambda version: selector_names.get(version, version), label_visibility="collapsed")
    selected = next(model for model in selector_models if model["model_version"] == selected_version)
    st.caption("Current survival model governs survival curves only; RUL uses the separate Current model. Validation selects candidate models; test metrics are held-out evaluation only.")
    flattened = pd.DataFrame([survival_model_metrics(model) for model in models])
    flattened.insert(0, "Display model", [model_names[model["model_version"]] for model in models])
    flattened = flattened.drop(columns=["Model version"], errors="ignore")
    flattened.insert(1, "Selection", ["Current" if model["model_version"] == selected_version else "" for model in models])
    st.subheader("Model performance comparison")
    st.dataframe(
        performance_gradient(
            flattened[["Display model", "Selection", "Generation", "Selected model family", "Validation IBS", "Validation IPCW C-index", "Test IBS", "Test IPCW C-index"]],
            lower_is_better=["Validation IBS", "Test IBS"],
            higher_is_better=["Validation IPCW C-index", "Test IPCW C-index"],
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Validation IBS": st.column_config.NumberColumn(format="%.5f"),
            "Validation IPCW C-index": st.column_config.NumberColumn(format="%.5f"),
            "Test IBS": st.column_config.NumberColumn(format="%.5f"),
            "Test IPCW C-index": st.column_config.NumberColumn(format="%.5f"),
        },
    )
    st.caption("Model selection uses validation metrics only; test metrics are held-out evaluation.")
    st.caption("IBS: how far predicted survival probabilities are from actual outcomes over time (↓ better)")
    st.caption("IPCW C-index: how well the model ranks which batteries are likely to fail sooner (↑ better)")
    st.subheader("Validation comparison")
    generation_version = st.selectbox(
        "Generation",
        [model["model_version"] for model in models],
        index=versions.index(selected_version),
        format_func=lambda version: model_names.get(version, version),
    )
    generation_model = next(model for model in models if model["model_version"] == generation_version)
    comparison = pd.DataFrame(survival_family_validation_rows(generation_model))
    if not comparison.empty:
        st.dataframe(
            performance_gradient(
                comparison,
                lower_is_better=["Validation IBS"],
                higher_is_better=["Validation IPCW C-index"],
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Validation IBS": st.column_config.NumberColumn(format="%.5f"),
                "Validation IPCW C-index": st.column_config.NumberColumn(format="%.5f"),
            },
        )


st.set_page_config(page_title="Battery reliability monitoring", layout="wide")
navigation = {
    "Battery Monitoring": fleet_page,
    "Battery Detail": battery_page,
    "RUL Model Monitoring": model_page,
    "Survival Model Monitoring": survival_model_page,
}
legacy_pages = {
    "Fleet monitoring": "Battery Monitoring",
    "Battery detail": "Battery Detail",
    "Model monitoring": "RUL Model Monitoring",
    "Survival model monitoring": "Survival Model Monitoring",
}
if "navigate_to" in st.session_state:
    st.session_state.page = st.session_state.pop("navigate_to")
if "page" in st.session_state:
    st.session_state.page = legacy_pages.get(st.session_state.page, st.session_state.page)
else:
    st.session_state.page = "Battery Monitoring"
st.sidebar.radio("", list(navigation), key="page", label_visibility="collapsed")
navigation[st.session_state.page]()
