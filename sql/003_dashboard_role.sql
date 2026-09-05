\if :{?dashboard_password}
SELECT format('CREATE ROLE analytics_dashboard LOGIN PASSWORD %L', :'dashboard_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analytics_dashboard')
\gexec
\endif

GRANT USAGE ON SCHEMA analytics TO analytics_dashboard;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_dashboard;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT ON TABLES TO analytics_dashboard;
GRANT INSERT, UPDATE ON analytics.current_models TO analytics_dashboard;
GRANT INSERT, UPDATE ON analytics.current_survival_models TO analytics_dashboard;
GRANT SELECT ON analytics.current_stream_states, analytics.stream_serving_status,
    analytics.battery_current_survival_predictions TO analytics_dashboard;
