"""Read-only Streamlit views over PostgreSQL analytics serving tables."""
import json
import os

import altair as alt
import pandas as pd
import psycopg
import streamlit as st
from psycopg.rows import dict_row

from src.dashboard_data import family_label, family_validation_rows, lifecycle_stage, latest_model_version, measured_soh_distribution, model_display_names, model_metrics, performance_gradient, selectable_models, soh_percent, survival_family_validation_rows, survival_model_metrics


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
            INSERT INTO analytics.current_models (dataset, model_version, selection_revision, updated_at)
            SELECT %(dataset)s, model_version, 1, NOW()
            FROM analytics.model_evaluations
            WHERE dataset = %(dataset)s AND model_version = %(model_version)s
              AND status IN ('candidate', 'champion')
            ON CONFLICT (dataset) DO UPDATE
            SET model_version = EXCLUDED.model_version,
                selection_revision = analytics.current_models.selection_revision + 1,
                updated_at = NOW()
            """,
            {"dataset": dataset, "model_version": model_version},
        )
    database().commit()


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
    models = selectable_models(models)
    if not models:
        return None
    versions = [model["model_version"] for model in models]
    selected = st.session_state.get("current_model_version")
    if selected not in versions:
        selected = latest_model_version(models)
        st.session_state.current_model_version = selected
    return next(model for model in models if model["model_version"] == selected)


def model_predictions(dataset, model_version):
    return pd.DataFrame(rows(
        """
        SELECT DISTINCT ON (battery_id) battery_id, predicted_rul_cycles,
               predicted_eol_cycle AS estimated_eol_cycle, prediction_created_at
        FROM analytics.battery_predictions
        WHERE dataset = %(dataset)s AND model_version = %(model_version)s
        ORDER BY battery_id, cycle_index DESC
        """,
        {"dataset": dataset, "model_version": model_version},
    ))


def current_model_predictions(dataset, model_version):
    # Keep merge keys available when the current serving table has not been scored.
    return pd.DataFrame(rows(
        """
        SELECT battery_id, predicted_rul_cycles, predicted_eol_cycle AS estimated_eol_cycle,
               inference_created_at AS prediction_created_at
        FROM analytics.battery_current_predictions
        WHERE dataset = %(dataset)s AND model_version = %(model_version)s
        """,
        {"dataset": dataset, "model_version": model_version},
    ), columns=["battery_id", "predicted_rul_cycles", "estimated_eol_cycle", "prediction_created_at"])


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
            """,
            {"dataset": dataset, "model_version": model_version},
        )
    database().commit()


def current_survival_model(models):
    if not models:
        return None
    versions = [model["model_version"] for model in models]
    persisted = rows("SELECT model_version FROM analytics.current_survival_models WHERE dataset = %(dataset)s", {"dataset": models[0]["dataset"]})
    selected = persisted[0]["model_version"] if persisted else st.session_state.get("current_survival_model_version", versions[0])
    if selected not in versions:
        selected = versions[0]
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

    st.subheader("Current model")
    active = selectable_models(models)
    versions = [model["model_version"] for model in active]
    if versions:
        st.selectbox(
            "Current model",
            versions,
            index=versions.index(selected_version) if selected_version in versions else 0,
            key="current_model_version",
            on_change=persist_current_model,
            args=(fleet.iloc[0]["dataset"],),
            format_func=lambda version: model_display_names(models).get(version, version),
            label_visibility="collapsed",
        )
    selected_model = current_model(models)
    selected_version = selected_model["model_version"] if selected_model else None
    selected_name = model_display_names(models).get(selected_version, "Unavailable")

    rul_predictions_available = 0
    if selected_version:
        prediction_frame = current_model_predictions(fleet.iloc[0]["dataset"], selected_version)
        fleet = fleet.drop(columns=["predicted_rul_cycles", "predicted_eol_cycle", "estimated_eol_cycle", "prediction_created_at"], errors="ignore").merge(prediction_frame, on="battery_id", how="left")
        rul_predictions_available = int(fleet["predicted_rul_cycles"].notna().sum())

    metric_row = st.columns(4)
    metric_row[0].metric("Batteries tracked", len(fleet))
    metric_row[1].metric("Average SOH", f"{fleet['measured_soh'].mean():.1%}")
    metric_row[2].metric("Median SOH", f"{fleet['measured_soh'].median():.1%}")
    metric_row[3].metric("RUL predictions available", rul_predictions_available)
    st.caption(f"Measured SOH is derived from capacity. RUL is predicted by {selected_name}.")

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
    if selected_version:
        filtered = filtered[filtered["predicted_rul_cycles"].fillna(float("inf")) <= max_rul]
    filtered = filtered.assign(lifecycle_stage=[lifecycle_stage(row.current_cycle, row.predicted_rul_cycles) for row in filtered.itertuples()])
    filtered["measured_soh_percent"] = filtered["measured_soh"].map(soh_percent)
    visible = filtered.rename(
        columns={
            "battery_id": "Battery",
            "current_cycle": "Current cycle",
            "measured_soh_percent": "Measured SOH (%)",
            "predicted_rul_cycles": "Predicted RUL (cycles)",
            "estimated_eol_cycle": "Estimated EOL cycle",
            "lifecycle_stage": "Lifecycle stage",
            "prediction_created_at": "Prediction timestamp",
        }
    )
    event = st.dataframe(
        visible[["Battery", "Current cycle", "Measured SOH (%)", "Predicted RUL (cycles)", "Estimated EOL cycle", "Lifecycle stage", "Prediction timestamp"]],
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        width="stretch",
        column_config={"Measured SOH (%)": st.column_config.NumberColumn(format="%.1f")},
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
    selected_prediction = rows(
        """
        SELECT predicted_rul_cycles, predicted_eol_cycle
        FROM analytics.battery_current_predictions
        WHERE model_version = %(model_version)s AND dataset = %(dataset)s AND battery_id = %(battery_id)s
        ORDER BY cycle_index DESC LIMIT 1
        """,
        {"model_version": selected_version, "dataset": latest.dataset, "battery_id": battery_id},
    ) if selected_version else []
    latest_rul = selected_prediction[0]["predicted_rul_cycles"] if selected_prediction else None
    latest_eol = selected_prediction[0]["predicted_eol_cycle"] if selected_prediction else None
    cards = st.columns(5)
    cards[0].metric("Current cycle", int(latest.current_cycle))
    cards[1].metric("Measured SOH", f"{latest.measured_soh:.1%}")
    cards[2].metric(f"Predicted RUL · {selected_name}", "Unavailable" if latest_rul is None else f"{latest_rul:.0f} cycles")
    cards[3].metric("Estimated EOL cycle", "Unavailable" if latest_eol is None else f"{latest_eol:.0f}")
    cards[4].metric("Lifecycle stage", lifecycle_stage(latest.current_cycle, latest_rul))
    st.caption("Measured SOH/capacity is independent of the selected model.")

    survival_models = survival_evaluations()
    selected_survival = current_survival_model(survival_models)
    serving = current_survival_serving_state(latest.dataset)
    if not serving:
        st.info("Finalized stream state is unavailable.")
    elif serving[0]["status"] != "served":
        status = serving[0]["status"] or "pending"
        detail = serving[0].get("error_message") if status == "failed" else None
        st.info(f"Survival serving is {status}." + (f" {detail}" if detail else ""))
    elif selected_survival:
        curve = pd.DataFrame(rows(
            """SELECT horizon_cycles, survival_probability FROM analytics.battery_current_survival_predictions
               WHERE model_version = %(model_version)s AND dataset = %(dataset)s AND battery_id = %(battery_id)s
                 AND state_id = %(state_id)s AND model_fingerprint = %(model_fingerprint)s
                 AND selection_revision = %(selection_revision)s ORDER BY cycle_index DESC, horizon_cycles""",
            {"model_version": selected_survival["model_version"], "dataset": latest.dataset, "battery_id": battery_id,
             "state_id": serving[0]["state_id"], "model_fingerprint": serving[0]["model_fingerprint"],
             "selection_revision": serving[0]["selection_revision"]},
        ))
        if not curve.empty:
            latest_curve = curve[curve["horizon_cycles"].notna()].drop_duplicates("horizon_cycles", keep="first").sort_values("horizon_cycles")
            st.subheader("Survival analysis")
            st.caption(f"{family_label(selected_survival['model_name'])}: conditional probability of remaining above the EOL threshold, given features at the current cycle.")
            st.line_chart(latest_curve.set_index("horizon_cycles")[["survival_probability"]])
            horizons = latest_curve.set_index("horizon_cycles")["survival_probability"]
            cards = st.columns(3)
            for card, horizon in zip(cards, (50, 100, 200)):
                card.metric(f"Survive +{horizon} cycles", "Unavailable" if horizon not in horizons else f"{horizons[horizon]:.1%}")

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
    selected_status = selected_model["status"]
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
    st.caption(f"{model_names.get(selected, selected)} · internal version: {selected}. Selected model status: {selected_status}.")
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
    active = selectable_models(models)
    model_names = model_display_names(active)
    if not active:
        st.info("No non-retired canonical model generations are available for monitoring.")
        return
    selected_model = current_model(models)
    versions = [model["model_version"] for model in active]
    if selected_model:
        st.selectbox(
            "Current model",
            versions,
            index=versions.index(selected_model["model_version"]) if selected_model and selected_model["model_version"] in versions else 0,
            key="current_model_version",
            on_change=persist_current_model,
            args=(selected_model["dataset"],),
            format_func=lambda version: model_names.get(version, version),
    )
    st.caption("The Current model is selected for dashboard analysis; database audit statuses are unchanged.")
    flattened = pd.DataFrame([model_metrics(model) for model in active])
    flattened.insert(0, "Display model", [model_names.get(model["model_version"], model["model_version"]) for model in active])
    flattened = flattened.rename(columns={"Model version": "Internal model version"})
    flattened.insert(1, "Selection", ["Current" if model.get("model_version") == st.session_state.get("current_model_version") else "" for model in active])
    st.dataframe(
        performance_gradient(
            flattened,
            lower_is_better=["Validation MAE", "Test MAE", "Test RMSE", "Early MAE", "Mid MAE", "Late MAE"],
            higher_is_better=["Test R²"],
        ),
        hide_index=True,
        width="stretch",
    )

    generation_version = st.selectbox(
        "Generation validation comparison",
        [model["model_version"] for model in active],
        format_func=lambda version: model_names.get(version, version),
    )
    generation_model = next(model for model in active if model["model_version"] == generation_version)
    family_validation = pd.DataFrame(family_validation_rows(generation_model))
    if family_validation.empty:
        st.info("This legacy evaluation does not contain family-level validation results.")
    else:
        st.caption("Family and configuration selection use validation data only. The selected family is marked below.")
        st.dataframe(
            performance_gradient(
                family_validation,
                lower_is_better=["Validation MAE", "Validation RMSE"],
                higher_is_better=["Validation R²"],
            ),
            hide_index=True,
            width="stretch",
        )
        st.altair_chart(
            alt.Chart(family_validation).mark_bar().encode(
                x=alt.X("Model family:N", sort=["Ridge", "Random Forest", "XGBoost", "MLP"]),
                y="Validation MAE:Q",
                color=alt.Color("Selected:N", scale=alt.Scale(domain=[False, True], range=["#9aa0a6", "#1f77b4"])),
                tooltip=["Model family", "Configuration", "Validation MAE", "Validation RMSE", "Validation R²", "Selected"],
            ),
            width="stretch",
        )

    st.subheader("Metric definitions")
    st.markdown(
        """- Test MAE — On average, how many cycles the prediction is off by; lower is better.
- Test RMSE — Similar to MAE, but gives more weight to large prediction errors; lower is better.
- Lifecycle MAE — Shows how prediction error changes from early to mid to late battery life; lower is better.
- R² — How well the model explains differences in remaining battery life; closer to 1 is better."""
    )

    if not active:
        st.info("No model is available for comparison.")
        return
    baseline = selected_model or active[-1]
    candidate_options = [model["model_version"] for model in active if model["model_version"] != baseline["model_version"]]
    selected_candidates = st.multiselect(
        "Add models to compare",
        options=candidate_options,
        default=[],
        format_func=lambda version: model_names.get(version, version),
        max_selections=4,
        help="The Current model is always included.",
    )
    selected_versions = [baseline["model_version"], *selected_candidates]
    selected_models = [model for model in active if model["model_version"] in selected_versions]
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
        heading = f"{display_name} ({model['status']})"
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


def survival_model_page():
    st.header("Survival model monitoring")
    all_models = survival_evaluations()
    models = selectable_models(all_models)
    if not models:
        st.info("No survival model evaluations are available.")
        return
    selected = current_survival_model(all_models)
    versions = [model["model_version"] for model in models]
    selected_version = selected["model_version"] if selected and selected["model_version"] in versions else versions[0]
    st.selectbox("Current survival model", versions, index=versions.index(selected_version), key="current_survival_model_version", on_change=persist_current_survival_model, args=(models[0]["dataset"],), format_func=lambda version: next(f"Model {model.get('generation')} — {family_label(model['model_name'])}" for model in models if model["model_version"] == version))
    st.caption("Survival models estimate conditional probabilities; they do not change RUL selection or metrics.")
    st.dataframe(
        performance_gradient(
            pd.DataFrame([survival_model_metrics(model) for model in models]),
            lower_is_better=["Validation IBS", "Test IBS"],
            higher_is_better=["Validation IPCW C-index", "Test IPCW C-index"],
        ),
        hide_index=True,
        width="stretch",
    )
    comparison_model = next((model for model in models if model["model_version"] == selected_version), models[0])
    comparison = pd.DataFrame(survival_family_validation_rows(comparison_model))
    if not comparison.empty:
        st.subheader("Family validation comparison")
        st.caption("Family and configuration selection use validation data only; the selected family is marked below.")
        st.dataframe(
            performance_gradient(
                comparison,
                lower_is_better=["Validation IBS"],
                higher_is_better=["Validation IPCW C-index"],
            ),
            hide_index=True,
            width="stretch",
        )
    test = json.loads(selected["metrics"]) if isinstance(selected["metrics"], str) else selected["metrics"]
    winner_test = test.get("test", {})
    st.subheader("Winner fixed-test metrics")
    winner_test_frame = pd.DataFrame([{
        "Model family": family_label(selected["model_name"]),
        "Test IBS": winner_test.get("integrated_brier_score"),
        "Test IPCW C-index": winner_test.get("ipcw_c_index"),
        "+50 Brier": winner_test.get("horizon_brier", {}).get("50"),
        "+100 Brier": winner_test.get("horizon_brier", {}).get("100"),
        "+200 Brier": winner_test.get("horizon_brier", {}).get("200"),
    }])
    st.dataframe(
        performance_gradient(
            winner_test_frame,
            lower_is_better=["Test IBS", "+50 Brier", "+100 Brier", "+200 Brier"],
            higher_is_better=["Test IPCW C-index"],
        ),
        hide_index=True,
        width="stretch",
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
