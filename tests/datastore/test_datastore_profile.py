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
from unittest.mock import patch

import pydantic.error_wrappers
import pytest

import mlrun
import mlrun.common.schemas
from mlrun.datastore.datastore_profile import (
    _DATASTORE_TYPE_TO_PROFILE_CLASS,
    DatastoreProfile,
    DatastoreProfile2Json,
    DatastoreProfileKafkaStream,
    DatastoreProfileKafkaTarget,
    DatastoreProfilePostgreSQL,
    DatastoreProfileTDEngine,
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


def test_kafka_stream_datastore() -> None:
    profile = DatastoreProfileKafkaStream(
        name="my_stream", topics=["my-topic"], brokers="localhost:9092"
    )
    assert profile.name == "my_stream"
    assert profile.get_topic() == "my-topic"
    assert profile.brokers == "localhost:9092"


@pytest.mark.parametrize(
    ("brokers_kwargs", "expected_err_msg"),
    [
        ({"brokers": None}, "none is not an allowed value"),
        ({}, "field required"),
    ],
)
@pytest.mark.parametrize(
    "profile_class", [DatastoreProfileKafkaTarget, DatastoreProfileKafkaStream]
)
def test_kafka_target_datastore_no_brokers(
    brokers_kwargs: dict, expected_err_msg: str, profile_class: type
) -> None:
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=expected_err_msg,
    ):
        if isinstance(profile_class, DatastoreProfileKafkaStream):
            profile_class(name="my_stream", topics=["my-topic"], **brokers_kwargs)
        else:
            profile_class(name="my_target", topic="my-topic", **brokers_kwargs)


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


class TestTDEngineProfile:
    @staticmethod
    def test_from_dsn() -> None:
        dsn = "taosws://root:taosdata@localhost:6041"
        profile_name = "test-taosws"
        profile = DatastoreProfileTDEngine.from_dsn(dsn=dsn, profile_name=profile_name)
        assert profile.type == "taosws"
        assert profile.user == "root"
        assert profile.password == "taosdata"
        assert profile.host == "localhost"
        assert profile.port == 6041
        assert (
            profile.dsn() == dsn
        ), "Converting the profile back to DSN did not work as expected"

    @staticmethod
    def test_datastore_profile_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
        profile_name = "test-profile"
        project_name = "test-project"

        public_profile = mlrun.common.schemas.DatastoreProfile(
            name=profile_name,
            type="taosws",
            object='{"type":"dGFvc3dz","name":"dGRlbmdpbmUx","user":"cm9vdA==","host":"MC4wLjAuMA==","port":"NjA0MQ=="}',
            private=None,
            project=project_name,
        )

        with patch(
            "mlrun.db.nopdb.NopDB.get_datastore_profile", return_value=public_profile
        ):
            monkeypatch.setenv(
                f"datastore-profiles.{project_name}.{profile_name}",
                '{"password": "MTIzNA=="}',
            )
            profile_read = datastore_profile_read(f"ds://{profile_name}", project_name)

        assert profile_read.type == "taosws", "Wrong profile type"
        assert profile_read.password == "1234", "Wrong password"


class TestDatastoreProfilePostgreSQL:
    @staticmethod
    def test_from_dsn() -> None:
        dsn = "postgresql://postgres:password123@localhost:5432/mydb"
        profile_name = "test-timescaledb"
        profile = DatastoreProfilePostgreSQL.from_dsn(
            dsn=dsn, profile_name=profile_name
        )
        assert profile.type == "postgresql"
        assert profile.user == "postgres"
        assert profile.password == "password123"
        assert profile.host == "localhost"
        assert profile.port == 5432
        assert profile.database == "mydb"
        assert (
            profile.dsn() == dsn
        ), "Converting the profile back to DSN did not work as expected"

    @staticmethod
    def test_from_dsn_without_database() -> None:
        dsn = "postgresql://postgres:password123@localhost:5432"
        profile_name = "test-timescaledb-no-db"
        profile = DatastoreProfilePostgreSQL.from_dsn(
            dsn=dsn, profile_name=profile_name
        )
        assert profile.type == "postgresql"
        assert profile.user == "postgres"
        assert profile.password == "password123"
        assert profile.host == "localhost"
        assert profile.port == 5432
        assert profile.database == "postgres"  # Should default to "postgres"

    @staticmethod
    def test_datastore_profile_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
        profile_name = "test-profile"
        project_name = "test-project"

        public_profile = mlrun.common.schemas.DatastoreProfile(
            name=profile_name,
            type="postgresql",
            object='{"type":"cG9zdGdyZXNxbA==","name":"dGltZXNjYWxlZGIx","user":"cG9zdGdyZXM=","host":"bG9jYWxob3N0","port":"NTQzMg==","database":"bXlkYg=="}',
            private=None,
            project=project_name,
        )

        with patch(
            "mlrun.db.nopdb.NopDB.get_datastore_profile", return_value=public_profile
        ):
            monkeypatch.setenv(
                f"datastore-profiles.{project_name}.{profile_name}",
                '{"password": "cGFzc3dvcmQxMjM="}',
            )
            profile_read = datastore_profile_read(f"ds://{profile_name}", project_name)

        assert profile_read.type == "postgresql", "Wrong profile type"
        assert profile_read.password == "password123", "Wrong password"


@pytest.fixture
def datastore_profile_classes() -> set[type[DatastoreProfile]]:
    subclasses = DatastoreProfile.__subclasses__()
    for subclass in subclasses:
        subclasses.extend(subclass.__subclasses__())
    return set(subclasses)


def test_datastore_type_map(
    datastore_profile_classes: set[type[DatastoreProfile]],
) -> None:
    assert (
        set(_DATASTORE_TYPE_TO_PROFILE_CLASS.values()) == datastore_profile_classes
    ), "Missing profiles in the map"
    for type_, profile_class in _DATASTORE_TYPE_TO_PROFILE_CLASS.items():
        assert type_ == profile_class.schema().get("properties", {}).get("type").get(
            "default"
        ), "Type key and profile class type do not match"
