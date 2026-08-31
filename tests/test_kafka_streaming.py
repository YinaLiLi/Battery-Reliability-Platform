import json

import pytest

from src.kafka_consumer import consume_messages
from src.kafka_producer import produce_rows


class FakeProducer:
    def __init__(self, callback_error=None):
        self.callback_error = callback_error
        self.produced = []
        self.poll_calls = 0
        self.flushed = False

    def produce(self, **kwargs):
        self.produced.append(kwargs)
        kwargs["on_delivery"](self.callback_error, None)

    def poll(self, timeout):
        self.poll_calls += 1

    def flush(self):
        self.flushed = True
        return 0


class FakeMessage:
    def __init__(self, value, partition=1, offset=4):
        self._value = value
        self._partition = partition
        self._offset = offset

    def error(self):
        return None

    def value(self):
        return self._value

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset


class FakeConsumer:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.committed = []

    def poll(self, timeout):
        return next(self.messages, None)

    def commit(self, message, asynchronous=False):
        self.committed.append((message, asynchronous))


def test_producer_limits_rows_and_flushes_after_asynchronous_delivery():
    producer = FakeProducer()

    sent = produce_rows(
        [{"vehicle_id": "EV-0001", "event_id": "one"}, {"vehicle_id": "EV-0002", "event_id": "two"}],
        producer,
        limit=1,
    )

    assert sent == 1
    assert producer.flushed
    assert producer.poll_calls == 1
    assert producer.produced[0]["key"] == b"EV-0001"
    assert json.loads(producer.produced[0]["value"]) == {"vehicle_id": "EV-0001", "event_id": "one"}


def test_producer_raises_after_flush_when_delivery_fails():
    producer = FakeProducer(callback_error=RuntimeError("broker unavailable"))

    with pytest.raises(RuntimeError, match="delivery failed"):
        produce_rows([{"vehicle_id": "EV-0001"}], producer, limit=0)

    assert producer.flushed


def test_consumer_prints_then_commits_each_successful_message():
    message = FakeMessage(json.dumps({"vehicle_id": "EV-0001"}).encode())
    consumer = FakeConsumer([message])
    printed = []

    processed = consume_messages(consumer, max_messages=1, output=printed.append)

    assert processed == 1
    assert consumer.committed == [(message, False)]
    assert json.loads(printed[0]) == {
        "offset": 4,
        "partition": 1,
        "value": {"vehicle_id": "EV-0001"},
    }
