from src.battery_events import event_from_measurement, ordered_measurements
from src.continuous_arrival import lifecycle_events_for_manifest, schedule_measurement


def test_event_contract_preserves_source_time_and_deterministic_key():
    row = {"event_id":"matr:b:2:3","dataset":"MATR","battery_id":"b","cycle_index":2,"sample_index":3,"source_time_in_s":9.0,"replay_event_time":"2020-01-02T00:00:09+00:00","voltage_in_V":3.2,"current_in_A":-1.0,"temperature_in_C":25.0,"charge_capacity_in_Ah":1.1,"discharge_capacity_in_Ah":1.0,"internal_resistance_in_ohm":None}
    event = event_from_measurement(row)
    assert event["schema_version"] == "1.0"
    assert event["source_time_in_s"] == 9.0
    assert event["replay_event_time"] != event["source_time_in_s"]
    assert "replay_sequence" in event


def test_ordered_measurements_uses_battery_cycle_sample_keys():
    rows = [{"battery_id":"b","cycle_index":2,"sample_index":0},{"battery_id":"a","cycle_index":2,"sample_index":1},{"battery_id":"a","cycle_index":1,"sample_index":9}]
    assert [(x['battery_id'],x['cycle_index'],x['sample_index']) for x in ordered_measurements(rows)] == [('a',1,9),('a',2,1),('b',2,0)]


def test_scheduled_measurement_preserves_source_time_and_lifecycle_events_are_separate():
    manifest = {
        "battery_id": "b",
        "dataset": "MATR",
        "start_time": "2020-01-03T00:00:00+00:00",
        "first_source_cycle": 1,
        "last_source_cycle": 4,
        "eol_cycle": 3,
        "valid_eol_label": True,
    }
    scheduled = schedule_measurement({"battery_id": "b", "cycle_index": 2, "sample_index": 1, "source_time_in_s": 9.0}, manifest, replay_sequence=7)

    assert scheduled["source_time_in_s"] == 9.0
    assert scheduled["replay_event_time"] == "2020-01-04T00:00:09+00:00"
    assert scheduled["replay_sequence"] == 7
    assert [event["event_type"] for event in lifecycle_events_for_manifest(manifest)] == ["eol_observed", "replay_complete"]
    assert lifecycle_events_for_manifest({**manifest, "valid_eol_label": False}) == [lifecycle_events_for_manifest(manifest)[1]]
