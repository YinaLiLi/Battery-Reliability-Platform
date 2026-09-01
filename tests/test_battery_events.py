from src.battery_events import event_from_measurement, ordered_measurements


def test_event_contract_preserves_source_time_and_deterministic_key():
    row = {"event_id":"matr:b:2:3","dataset":"MATR","battery_id":"b","cycle_index":2,"sample_index":3,"source_time_in_s":9.0,"replay_event_time":"2020-01-02T00:00:09+00:00","voltage_in_V":3.2,"current_in_A":-1.0,"temperature_in_C":25.0,"charge_capacity_in_Ah":1.1,"discharge_capacity_in_Ah":1.0,"internal_resistance_in_ohm":None}
    event = event_from_measurement(row)
    assert event["schema_version"] == "1.0"
    assert event["source_time_in_s"] == 9.0
    assert event["replay_event_time"] != event["source_time_in_s"]


def test_ordered_measurements_uses_battery_cycle_sample_keys():
    rows = [{"battery_id":"b","cycle_index":2,"sample_index":0},{"battery_id":"a","cycle_index":2,"sample_index":1},{"battery_id":"a","cycle_index":1,"sample_index":9}]
    assert [(x['battery_id'],x['cycle_index'],x['sample_index']) for x in ordered_measurements(rows)] == [('a',1,9),('a',2,1),('b',2,0)]
