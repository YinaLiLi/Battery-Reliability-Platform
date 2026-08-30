# EV Fleet Reliability

## Project Objective

Build an end-to-end EV fleet telemetry pipeline to predict battery failure risk using simulated fleet data.

## Telemetry Schema

### Vehicle metadata
- vehicle_id
- battery_age_days
- battery_type
- region

### Time-series telemetry
- vehicle_id
- timestamp
- soc
- pack_voltage
- pack_current
- module_temp_min
- module_temp_max
- outside_temp
- odometer
- is_charging

## Data Sources

The synthetic fleet telemetry schema is informed by Tesla's publicly documented Fleet Telemetry fields. Public battery datasets, including NASA battery aging data, will be used as references for realistic battery behavior and degradation patterns.