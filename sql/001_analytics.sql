CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.battery_cycle_health (
    dataset TEXT NOT NULL, battery_id TEXT NOT NULL, cycle_index INTEGER NOT NULL CHECK (cycle_index > 0),
    soh DOUBLE PRECISION NOT NULL CHECK (soh >= 0), rul_cycles INTEGER,
    discharge_capacity_in_ah DOUBLE PRECISION, internal_resistance_in_ohm DOUBLE PRECISION,
    temperature_max_in_c DOUBLE PRECISION, charge_time_in_s DOUBLE PRECISION,
    capacity_slope_10 DOUBLE PRECISION, coulombic_efficiency DOUBLE PRECISION,
    PRIMARY KEY (dataset, battery_id, cycle_index)
);
CREATE INDEX IF NOT EXISTS battery_cycle_health_battery_cycle_idx ON analytics.battery_cycle_health (battery_id, cycle_index);

CREATE TABLE IF NOT EXISTS analytics.battery_predictions (
    model_version TEXT NOT NULL, dataset TEXT NOT NULL, battery_id TEXT NOT NULL, cycle_index INTEGER NOT NULL CHECK (cycle_index > 0),
    predicted_rul_cycles DOUBLE PRECISION NOT NULL, prediction_created_at TIMESTAMPTZ NOT NULL, split TEXT NOT NULL,
    PRIMARY KEY (model_version, dataset, battery_id, cycle_index)
);
CREATE INDEX IF NOT EXISTS battery_predictions_battery_cycle_idx ON analytics.battery_predictions (battery_id, cycle_index);

CREATE TABLE IF NOT EXISTS analytics.battery_replay_windows (
    battery_id TEXT NOT NULL, cycle_index INTEGER NOT NULL CHECK (cycle_index > 0), event_count BIGINT NOT NULL CHECK (event_count > 0),
    average_voltage_in_v DOUBLE PRECISION, maximum_temperature_in_c DOUBLE PRECISION,
    charge_capacity_in_ah DOUBLE PRECISION, discharge_capacity_in_ah DOUBLE PRECISION, internal_resistance_in_ohm DOUBLE PRECISION,
    PRIMARY KEY (battery_id, cycle_index)
);

CREATE TABLE IF NOT EXISTS analytics.model_evaluations (
    model_version TEXT PRIMARY KEY, model_name TEXT NOT NULL, dataset TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('candidate', 'champion')),
    evaluated_at TIMESTAMPTZ NOT NULL, metrics JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS model_evaluations_status_idx ON analytics.model_evaluations (status, evaluated_at DESC);
