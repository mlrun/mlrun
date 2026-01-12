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
    DatastoreProfileRedis,
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
    def test_url_encoding_special_characters() -> None:
        """Test that special characters in user/password are properly URL encoded/decoded."""
        profile = DatastoreProfilePostgreSQL(
            name="test-profile",
            user="user@domain",
            password="p@ss:word/123",
            host="localhost",
            port=5432,
            database="mydb",
        )
        dsn = profile.dsn()
        assert (
            dsn == "postgresql://user%40domain:p%40ss%3Aword%2F123@localhost:5432/mydb"
        )

        # Round-trip: from_dsn should decode back to original values
        restored = DatastoreProfilePostgreSQL.from_dsn(dsn=dsn, profile_name="restored")
        assert restored.user == "user@domain"
        assert restored.password == "p@ss:word/123"
        assert restored.database == "mydb"

    @staticmethod
    def test_url_encoding_round_trip() -> None:
        """Test that from_dsn -> dsn produces consistent results."""
        # Start with an encoded DSN
        encoded_dsn = "postgresql://user%40domain:p%40ss%3Aword@localhost:5432/mydb"
        profile = DatastoreProfilePostgreSQL.from_dsn(
            dsn=encoded_dsn, profile_name="test"
        )

        # Verify decoded values
        assert profile.user == "user@domain"
        assert profile.password == "p@ss:word"

        # Round-trip should produce the same DSN
        assert profile.dsn() == encoded_dsn

    @staticmethod
    def test_url_encoding_database_with_special_chars() -> None:
        """Test that database names with special characters are properly encoded."""
        profile = DatastoreProfilePostgreSQL(
            name="test-profile",
            user="postgres",
            password="password",
            host="localhost",
            port=5432,
            database="my/db?name",
        )
        dsn = profile.dsn()
        assert dsn == "postgresql://postgres:password@localhost:5432/my%2Fdb%3Fname"

        # Round-trip should preserve the database name
        restored = DatastoreProfilePostgreSQL.from_dsn(dsn=dsn, profile_name="restored")
        assert restored.database == "my/db?name"

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
    def test_dsn_with_database_override() -> None:
        """Test dsn() method with database parameter override."""
        profile = DatastoreProfilePostgreSQL(
            name="test-profile",
            user="postgres",
            password="password123",
            host="localhost",
            port=5432,
            database="mydb",
        )
        # Default behavior - use configured database
        assert profile.dsn() == "postgresql://postgres:password123@localhost:5432/mydb"

        # Override with different database
        assert (
            profile.dsn(database="otherdb")
            == "postgresql://postgres:password123@localhost:5432/otherdb"
        )

        # Original database is unchanged
        assert profile.database == "mydb"

    @staticmethod
    def test_admin_dsn() -> None:
        """Test admin_dsn() returns DSN pointing to default postgres database."""
        profile = DatastoreProfilePostgreSQL(
            name="test-profile",
            user="postgres",
            password="password123",
            host="localhost",
            port=5432,
            database="mydb",
        )
        # admin_dsn should point to 'postgres' database for administrative operations
        assert (
            profile.admin_dsn()
            == "postgresql://postgres:password123@localhost:5432/postgres"
        )

        # Original database is unchanged
        assert profile.database == "mydb"

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


class TestDatastoreProfileRedis:
    @staticmethod
    def test_url_with_credentials_basic() -> None:
        """Test url_with_credentials with simple credentials."""
        profile = DatastoreProfileRedis(
            name="test-redis",
            endpoint_url="redis://localhost:6379/0",
            username="user",
            password="pass",
        )
        url = profile.url_with_credentials()
        assert url == "redis://user:pass@localhost:6379/0"

    @staticmethod
    def test_url_with_credentials_special_characters() -> None:
        """Test that special characters in username/password are properly URL encoded."""
        profile = DatastoreProfileRedis(
            name="test-redis",
            endpoint_url="redis://localhost:6379/0",
            username="user@domain",
            password="p@ss:word/123",
        )
        assert (
            profile.url_with_credentials()
            == "redis://user%40domain:p%40ss%3Aword%2F123@localhost:6379/0"
        )

    @staticmethod
    def test_url_with_credentials_username_only() -> None:
        """Test url_with_credentials with username only (no password)."""
        profile = DatastoreProfileRedis(
            name="test-redis",
            endpoint_url="redis://localhost:6379/0",
            username="user@domain",
            password=None,
        )
        url = profile.url_with_credentials()
        assert url == "redis://user%40domain@localhost:6379/0"

    @staticmethod
    def test_url_with_credentials_no_credentials() -> None:
        """Test url_with_credentials without any credentials."""
        profile = DatastoreProfileRedis(
            name="test-redis",
            endpoint_url="redis://localhost:6379/0",
            username=None,
            password=None,
        )
        url = profile.url_with_credentials()
        assert url == "redis://localhost:6379/0"


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
