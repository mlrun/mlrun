import pytest

import mlrun.errors
from mlrun.datastore.datastore_profile import DatastoreProfileKafkaTarget


def test_kafka_target_datastore():
    profile = DatastoreProfileKafkaTarget(
        name="my_target", topic="my-topic", brokers="localhost:9092"
    )
    assert profile.name == "my_target"
    assert profile.topic == "my-topic"
    assert profile.brokers == "localhost:9092"
    assert profile.bootstrap_servers is None


def test_kafka_target_datastore_bootstrap_servers_bwc():
    with pytest.warns(
        FutureWarning,
        match="'bootstrap_servers' parameter is deprecated in 1.7.0 "
        "and will be removed in 1.9.0, use 'brokers' instead.",
    ):
        profile = DatastoreProfileKafkaTarget(
            name="my_target", topic="my-topic", bootstrap_servers="localhost:9092"
        )
    assert profile.name == "my_target"
    assert profile.topic == "my-topic"
    assert profile.brokers == "localhost:9092"
    assert profile.bootstrap_servers is None


def test_kafka_target_datastore_no_brokers():
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="DatastoreProfileKafkaTarget requires the 'brokers' field to be set",
    ):
        DatastoreProfileKafkaTarget(name="my_target", topic="my-topic")


def test_kafka_target_datastore_brokers_and_bootstrap_servers():
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="DatastoreProfileKafkaTarget cannot be created with both 'brokers' and 'bootstrap_servers'",
    ):
        DatastoreProfileKafkaTarget(
            name="my_target",
            topic="my-topic",
            brokers="localhost:9092",
            bootstrap_servers="localhost:9092",
        )
