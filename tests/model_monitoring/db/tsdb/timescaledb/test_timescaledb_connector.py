# Copyright 2025 Iguazio
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

"""Tests for TimescaleDBConnector database auto-creation functionality."""

import contextlib
import uuid

import pytest

import mlrun
from mlrun.utils import logger


class TestTimescaleDBConnectorDatabaseCreation:
    """Test database auto-creation functionality in TimescaleDBConnector."""

    @pytest.fixture
    def test_system_id(self):
        """Generate unique system_id for test isolation."""
        return f"test_{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def profile(self, connection_string):
        """Create profile from connection string."""
        from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL

        return DatastoreProfilePostgreSQL.from_dsn(
            dsn=connection_string, profile_name="test_profile"
        )

    @staticmethod
    def _force_drop_database(admin_connection, database_name):
        """Force drop a database by terminating all connections first.

        Logs warnings on errors but doesn't fail tests.
        """
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )

        # Terminate all connections to the database
        try:
            admin_connection.run(
                query=Statement(
                    sql="""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    parameters=(database_name,),
                )
            )
        except Exception as e:
            logger.warning(
                "Failed to terminate connections during test cleanup",
                database=database_name,
                error=str(e),
            )

        # Now drop the database
        try:
            admin_connection.run(
                statements=[f'DROP DATABASE IF EXISTS "{database_name}"']
            )
        except Exception as e:
            logger.warning(
                "Failed to drop database during test cleanup",
                database=database_name,
                error=str(e),
            )

    @staticmethod
    @contextlib.contextmanager
    def _config_context(system_id=None, auto_create=None):
        """Context manager that saves and restores mlrun config.

        Always restores config even if test fails.
        """
        original_system_id = mlrun.mlconf.system_id
        original_auto_create = (
            mlrun.mlconf.model_endpoint_monitoring.tsdb.auto_create_database
        )

        try:
            if system_id is not None:
                mlrun.mlconf.system_id = system_id
            if auto_create is not None:
                mlrun.mlconf.model_endpoint_monitoring.tsdb.auto_create_database = (
                    auto_create
                )
            yield
        finally:
            mlrun.mlconf.system_id = original_system_id
            mlrun.mlconf.model_endpoint_monitoring.tsdb.auto_create_database = (
                original_auto_create
            )

    def test_auto_create_database_creates_new_database(
        self, project_name, test_system_id, profile, admin_connection
    ):
        """Test that auto_create_database=True creates a new database."""
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connector import (
            TimescaleDBConnector,
        )

        expected_db_name = f"mlrun_mm_{test_system_id}"

        with self._config_context(system_id=test_system_id, auto_create=True):
            connector = TimescaleDBConnector(
                project=project_name,
                profile=profile,
            )
            try:
                connector.create_tables()

                # Verify database was created by querying pg_database
                result = admin_connection.run(
                    query=Statement(
                        sql="SELECT 1 FROM pg_database WHERE datname = %s",
                        parameters=(expected_db_name,),
                    )
                )
                assert result.data, f"Database {expected_db_name} should exist"

            finally:
                # Cleanup resources - ignore errors
                with contextlib.suppress(Exception):
                    connector.delete_tsdb_resources()
                self._force_drop_database(admin_connection, expected_db_name)

    def test_auto_create_database_disabled_uses_existing_database(
        self, project_name, profile
    ):
        """Test that auto_create_database=False uses database from connection string."""
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connector import (
            TimescaleDBConnector,
        )

        connector = None

        with self._config_context(auto_create=False):
            try:
                # Create connector - should use database from profile
                connector = TimescaleDBConnector(
                    project=project_name,
                    profile=profile,
                )

                # Verify the connector uses the database from the profile
                # The profile should not be modified when auto_create is disabled
                assert profile.database == "postgres"

                connector.create_tables()

            finally:
                # Cleanup resources - ignore errors
                if connector:
                    with contextlib.suppress(Exception):
                        connector.delete_tsdb_resources()

    def test_auto_create_database_without_system_id_raises_error(
        self, project_name, profile
    ):
        """Test that auto_create_database=True without system_id raises error."""
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connector import (
            TimescaleDBConnector,
        )

        with self._config_context(system_id="", auto_create=True):
            # Should raise error - no cleanup needed since connector creation fails
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError, match="system_id is not set"
            ):
                TimescaleDBConnector(
                    project=project_name,
                    profile=profile,
                )

    def test_auto_create_database_idempotent(
        self, project_name, test_system_id, profile, admin_connection
    ):
        """Test that creating database multiple times is idempotent."""
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
            Statement,
        )
        from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connector import (
            TimescaleDBConnector,
        )

        expected_db_name = f"mlrun_mm_{test_system_id}"
        connector1 = None
        connector2 = None

        with self._config_context(system_id=test_system_id, auto_create=True):
            try:
                # Create first connector
                connector1 = TimescaleDBConnector(
                    project=project_name,
                    profile=profile,
                )
                connector1.create_tables()

                # Create second connector with same system_id - should not fail
                connector2 = TimescaleDBConnector(
                    project=f"{project_name}-2",
                    profile=profile,
                )
                connector2.create_tables()

                # Both should work - verify database exists
                result = admin_connection.run(
                    query=Statement(
                        sql="SELECT 1 FROM pg_database WHERE datname = %s",
                        parameters=(expected_db_name,),
                    )
                )
                assert result.data, f"Database {expected_db_name} should exist"

            finally:
                # Cleanup resources - ignore errors
                if connector1:
                    with contextlib.suppress(Exception):
                        connector1.delete_tsdb_resources()
                if connector2:
                    with contextlib.suppress(Exception):
                        connector2.delete_tsdb_resources()
                self._force_drop_database(admin_connection, expected_db_name)
