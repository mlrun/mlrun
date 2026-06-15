# Copyright 2026 Iguazio
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

from unittest.mock import patch

import pytest

import mlrun
from mlrun.common.model_monitoring.helpers import TIMESCALEDB_DEFAULT_DB_PREFIX
from mlrun.datastore.datastore_profile import (
    DatastoreProfilePostgreSQL,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)


class TestTimescaleDBStoreyTargetDatabaseResolution:
    """Tests for TimescaleDBStoreyTarget database name resolution.

    These tests verify that TimescaleDBStoreyTarget uses the shared
    get_tsdb_database_name() helper to ensure consistent database naming
    with TimescaleDBConnector.
    """

    @staticmethod
    @pytest.fixture
    def postgresql_profile() -> DatastoreProfilePostgreSQL:
        return DatastoreProfilePostgreSQL(
            name="test_tsdb_profile",
            user="testuser",
            password="testpass",
            host="localhost",
            port="5432",
            database="postgres",
        )

    @staticmethod
    @pytest.fixture
    def registered_profile(
        postgresql_profile: DatastoreProfilePostgreSQL,
    ):
        register_temporary_client_datastore_profile(postgresql_profile)
        yield postgresql_profile
        remove_temporary_client_datastore_profile(postgresql_profile.name)

    @staticmethod
    def test_ds_url_uses_auto_generated_database_name(
        monkeypatch: pytest.MonkeyPatch,
        registered_profile: DatastoreProfilePostgreSQL,
    ) -> None:
        system_id = "test_system_123"
        monkeypatch.setattr(
            mlrun.mlconf.model_endpoint_monitoring.tsdb,
            "auto_create_database",
            True,
        )
        monkeypatch.setattr(mlrun.mlconf, "system_id", system_id)

        expected_database = f"{TIMESCALEDB_DEFAULT_DB_PREFIX}_{system_id}"
        expected_dsn = registered_profile.dsn(database=expected_database)

        with patch("storey.TimescaleDBTarget.__init__", return_value=None) as mock_init:
            from mlrun.datastore.storeytargets import TimescaleDBStoreyTarget

            TimescaleDBStoreyTarget(url=f"ds://{registered_profile.name}")

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["dsn"] == expected_dsn

    @staticmethod
    def test_ds_url_uses_profile_database_when_auto_create_disabled(
        monkeypatch: pytest.MonkeyPatch,
        registered_profile: DatastoreProfilePostgreSQL,
    ) -> None:
        monkeypatch.setattr(
            mlrun.mlconf.model_endpoint_monitoring.tsdb,
            "auto_create_database",
            False,
        )

        expected_dsn = registered_profile.dsn()

        with patch("storey.TimescaleDBTarget.__init__", return_value=None) as mock_init:
            from mlrun.datastore.storeytargets import TimescaleDBStoreyTarget

            TimescaleDBStoreyTarget(url=f"ds://{registered_profile.name}")

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["dsn"] == expected_dsn

    @staticmethod
    def test_non_ds_url_passes_through_unchanged(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        direct_dsn = "postgresql://user:pass@host:5432/mydb"

        with patch("storey.TimescaleDBTarget.__init__", return_value=None) as mock_init:
            from mlrun.datastore.storeytargets import TimescaleDBStoreyTarget

            TimescaleDBStoreyTarget(url=direct_dsn)

            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["dsn"] == direct_dsn

    @staticmethod
    def test_non_postgresql_profile_raises_error(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from mlrun.datastore.datastore_profile import DatastoreProfileV3io

        v3io_profile = DatastoreProfileV3io(name="v3io_profile")
        register_temporary_client_datastore_profile(v3io_profile)

        try:
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Only DatastoreProfilePostgreSQL is supported",
            ):
                from mlrun.datastore.storeytargets import TimescaleDBStoreyTarget

                TimescaleDBStoreyTarget(url=f"ds://{v3io_profile.name}")
        finally:
            remove_temporary_client_datastore_profile(v3io_profile.name)
