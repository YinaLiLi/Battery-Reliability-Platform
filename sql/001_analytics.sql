CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.vehicle_features (
    event_id TEXT PRIMARY KEY,
    vehicle_id TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    battery_age_days INTEGER,
    battery_type TEXT,
    region TEXT,
    soc DOUBLE PRECISION,
    pack_voltage DOUBLE PRECISION,
    pack_current DOUBLE PRECISION,
    module_temp_min DOUBLE PRECISION,
    module_temp_max DOUBLE PRECISION,
    outside_temp DOUBLE PRECISION,
    odometer DOUBLE PRECISION,
    is_charging BOOLEAN,
    module_temp_spread DOUBLE PRECISION,
    previous_pack_voltage DOUBLE PRECISION,
    rolling_module_temp_max DOUBLE PRECISION,
    failure_within_30_operating_days SMALLINT NOT NULL CHECK (failure_within_30_operating_days IN (0, 1)),
    vehicle_battery_type TEXT,
    vehicle_region TEXT,
    CHECK (module_temp_min <= module_temp_max)
);

CREATE INDEX IF NOT EXISTS vehicle_features_region_event_time_idx
    ON analytics.vehicle_features (region, event_time);
CREATE INDEX IF NOT EXISTS vehicle_features_vehicle_event_time_idx
    ON analytics.vehicle_features (vehicle_id, event_time);

CREATE TABLE IF NOT EXISTS analytics.vehicle_window_metrics (
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    vehicle_id TEXT NOT NULL,
    event_count BIGINT NOT NULL CHECK (event_count > 0),
    average_pack_voltage DOUBLE PRECISION,
    maximum_module_temperature DOUBLE PRECISION,
    PRIMARY KEY (window_start, window_end, vehicle_id),
    CHECK (window_start < window_end)
);

CREATE INDEX IF NOT EXISTS vehicle_window_metrics_vehicle_start_idx
    ON analytics.vehicle_window_metrics (vehicle_id, window_start);
