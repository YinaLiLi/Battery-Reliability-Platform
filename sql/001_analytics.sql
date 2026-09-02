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
    raw_predicted_rul_cycles DOUBLE PRECISION, predicted_rul_cycles DOUBLE PRECISION NOT NULL CHECK (predicted_rul_cycles >= 0), predicted_eol_cycle DOUBLE PRECISION, prediction_created_at TIMESTAMPTZ NOT NULL, split TEXT NOT NULL,
    PRIMARY KEY (model_version, dataset, battery_id, cycle_index)
);
CREATE INDEX IF NOT EXISTS battery_predictions_battery_cycle_idx ON analytics.battery_predictions (battery_id, cycle_index);

ALTER TABLE analytics.battery_predictions
    ADD COLUMN IF NOT EXISTS raw_predicted_rul_cycles DOUBLE PRECISION;
ALTER TABLE analytics.battery_predictions
    ADD COLUMN IF NOT EXISTS predicted_eol_cycle DOUBLE PRECISION;
UPDATE analytics.battery_predictions
SET raw_predicted_rul_cycles = predicted_rul_cycles
WHERE raw_predicted_rul_cycles IS NULL;
UPDATE analytics.battery_predictions
SET predicted_rul_cycles = GREATEST(0, predicted_rul_cycles)
WHERE predicted_rul_cycles < 0;
WITH eol AS (
    SELECT model_version, dataset, battery_id, cycle_index,
        MIN(cycle_index) FILTER (WHERE predicted_rul_cycles = 0) OVER (PARTITION BY model_version, dataset, battery_id) AS first_eol_cycle
    FROM analytics.battery_predictions
)
UPDATE analytics.battery_predictions AS prediction
SET predicted_rul_cycles = CASE WHEN eol.first_eol_cycle IS NOT NULL AND prediction.cycle_index >= eol.first_eol_cycle THEN 0 ELSE prediction.predicted_rul_cycles END,
    predicted_eol_cycle = CASE
        WHEN eol.first_eol_cycle IS NOT NULL AND prediction.cycle_index >= eol.first_eol_cycle THEN eol.first_eol_cycle
        ELSE prediction.cycle_index + prediction.predicted_rul_cycles
    END
FROM eol
WHERE (prediction.model_version, prediction.dataset, prediction.battery_id, prediction.cycle_index) = (eol.model_version, eol.dataset, eol.battery_id, eol.cycle_index);
ALTER TABLE analytics.battery_predictions
    ALTER COLUMN predicted_eol_cycle SET NOT NULL;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'battery_predictions_nonnegative_rul') THEN
        ALTER TABLE analytics.battery_predictions
            ADD CONSTRAINT battery_predictions_nonnegative_rul CHECK (predicted_rul_cycles >= 0);
    END IF;
END $$;
CREATE TABLE IF NOT EXISTS analytics.battery_replay_windows (
    battery_id TEXT NOT NULL, cycle_index INTEGER NOT NULL CHECK (cycle_index > 0), event_count BIGINT NOT NULL CHECK (event_count > 0),
    average_voltage_in_v DOUBLE PRECISION, maximum_temperature_in_c DOUBLE PRECISION,
    charge_capacity_in_ah DOUBLE PRECISION, discharge_capacity_in_ah DOUBLE PRECISION, internal_resistance_in_ohm DOUBLE PRECISION,
    PRIMARY KEY (battery_id, cycle_index)
);

CREATE TABLE IF NOT EXISTS analytics.model_evaluations (
    model_version TEXT PRIMARY KEY, model_name TEXT NOT NULL, dataset TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('candidate', 'champion', 'retired')),
    evaluated_at TIMESTAMPTZ NOT NULL, metrics JSONB NOT NULL,
    training_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_fingerprint TEXT,
    generation TEXT
);
CREATE INDEX IF NOT EXISTS model_evaluations_status_idx ON analytics.model_evaluations (status, evaluated_at DESC);

ALTER TABLE analytics.model_evaluations
    ADD COLUMN IF NOT EXISTS training_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE analytics.model_evaluations
    ADD COLUMN IF NOT EXISTS model_fingerprint TEXT;
ALTER TABLE analytics.model_evaluations
    ADD COLUMN IF NOT EXISTS generation INTEGER;
ALTER TABLE analytics.model_evaluations
    ALTER COLUMN generation TYPE TEXT USING generation::TEXT;
ALTER TABLE analytics.model_evaluations DROP CONSTRAINT IF EXISTS model_evaluations_status_check;
ALTER TABLE analytics.model_evaluations
    ADD CONSTRAINT model_evaluations_status_check CHECK (status IN ('candidate', 'champion', 'retired'));
CREATE UNIQUE INDEX IF NOT EXISTS model_evaluations_fingerprint_idx
    ON analytics.model_evaluations (model_fingerprint) WHERE model_fingerprint IS NOT NULL;

CREATE INDEX IF NOT EXISTS battery_cycle_health_latest_idx
    ON analytics.battery_cycle_health (dataset, battery_id, cycle_index DESC);

CREATE UNIQUE INDEX IF NOT EXISTS model_evaluations_one_champion_per_dataset_idx
    ON analytics.model_evaluations (dataset) WHERE status = 'champion';

DROP VIEW IF EXISTS analytics.dashboard_battery_latest;
CREATE VIEW analytics.dashboard_battery_latest AS
WITH latest_health AS (
    SELECT DISTINCT ON (dataset, battery_id)
        dataset, battery_id, cycle_index, soh, discharge_capacity_in_ah,
        internal_resistance_in_ohm, temperature_max_in_c, capacity_slope_10
    FROM analytics.battery_cycle_health
    ORDER BY dataset, battery_id, cycle_index DESC
), champion AS (
    SELECT dataset, model_version
    FROM analytics.model_evaluations
    WHERE status = 'champion'
)
SELECT
    health.dataset,
    health.battery_id,
    health.cycle_index AS current_cycle,
    health.soh AS measured_soh,
    health.discharge_capacity_in_ah AS measured_capacity_in_ah,
    health.internal_resistance_in_ohm,
    health.temperature_max_in_c,
    health.capacity_slope_10,
    champion.model_version AS champion_model_version,
    prediction.predicted_rul_cycles AS predicted_rul_cycles,
    prediction.predicted_eol_cycle,
    prediction.prediction_created_at,
    prediction.predicted_eol_cycle AS estimated_eol_cycle
FROM latest_health AS health
LEFT JOIN champion ON champion.dataset = health.dataset
LEFT JOIN analytics.battery_predictions AS prediction
    ON prediction.model_version = champion.model_version
    AND prediction.dataset = health.dataset
    AND prediction.battery_id = health.battery_id
    AND prediction.cycle_index = health.cycle_index;
