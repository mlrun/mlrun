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

from unittest.mock import Mock

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.db.tsdb.preaggregate import PreAggregateConfig
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connector import (
    TimescaleDBConnector,
)


@pytest.fixture
def mock_profile():
    """Create a mock datastore profile."""
    profile = Mock(spec=DatastoreProfilePostgreSQL)
    profile.name = "test_profile"
    profile.dsn.return_value = "postgresql://user:pass@localhost:5432/test_db"
    return profile


# Note: pre_aggregate_config fixture is now available from conftest.py


class TestTimescaleDBConnector:
    """Test TimescaleDBConnector implementation."""

    def test_can_instantiate_connector(self, mock_profile):
        """Test that TimescaleDBConnector can be instantiated (all abstract methods implemented)."""
        # If any abstract methods are missing, this will fail
        connector = TimescaleDBConnector(
            project="test_project",
            profile=mock_profile,
        )

        assert isinstance(connector, TSDBConnector)
        assert connector.type == mm_schemas.TSDBTarget.TimescaleDB
        assert connector.project == "test_project"

    def test_can_instantiate_with_pre_aggregates(
        self, mock_profile, pre_aggregate_config
    ):
        """Test instantiation with pre-aggregate configuration."""
        connector = TimescaleDBConnector(
            project="test_project",
            profile=mock_profile,
            pre_aggregate_config=pre_aggregate_config,
        )

        assert isinstance(connector, TSDBConnector)
        config = connector.get_preaggregate_config()
        assert config == pre_aggregate_config

    def test_get_preaggregate_config_method(self, mock_profile):
        """Test the new get_preaggregate_config method."""
        # Without config
        connector = TimescaleDBConnector(
            project="test_project",
            profile=mock_profile,
            pre_aggregate_config=None,
        )
        assert connector.get_preaggregate_config() is None

        # With config
        config = PreAggregateConfig()
        connector = TimescaleDBConnector(
            project="test_project",
            profile=mock_profile,
            pre_aggregate_config=config,
        )
        assert connector.get_preaggregate_config() == config
