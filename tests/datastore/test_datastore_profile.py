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

from collections.abc import Iterator

import pytest

import mlrun.common.schemas
import mlrun.errors
from mlrun.datastore.datastore_profile import (
    DatastoreProfile2Json,
    DatastoreProfileKafkaTarget,
    DatastoreProfileV3io,
    datastore_profile_read,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)


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


@pytest.fixture
def v3io_profile_name() -> Iterator[str]:
    profile_name = "temp-prof"
    profile = DatastoreProfileV3io(name=profile_name)
    register_temporary_client_datastore_profile(profile)
    yield f"ds://{profile_name}"
    remove_temporary_client_datastore_profile(profile_name)


def test_temp_v3io_profile(v3io_profile_name: str) -> None:
    profile = datastore_profile_read(v3io_profile_name)
    assert profile.type == "v3io", "Wrong profile type"


def test_from_public_json() -> None:
    public_profile_schema = mlrun.common.schemas.DatastoreProfile(
        name="mm-infra-tsdb",
        type="v3io",
        object='{"type":"djNpbw==","name":"bW0taW5mcmEtc3RyZWFt"}',
        private=None,
        project="proj-11",
    )
    profile = DatastoreProfile2Json.create_from_json(public_profile_schema.object)
    assert isinstance(profile, DatastoreProfileV3io), "Not the right profile"
