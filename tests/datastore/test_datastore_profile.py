# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
