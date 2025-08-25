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

from typing import Optional

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
from mlrun.model_monitoring.db.tsdb.preaggregate import (
    PreAggregateHandler,
)

# Import the mixin classes
from mlrun.model_monitoring.db.tsdb.timescaledb.queries.timescaledb_metrics_queries import (
    TimescaleDBMetricsQueries,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.queries.timescaledb_predictions_queries import (
    TimescaleDBPredictionsQueries,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.queries.timescaledb_results_queries import (
    TimescaleDBResultsQueries,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    TimescaleDBConnection,
)


class TimescaleDBQueryHandler(
    TimescaleDBMetricsQueries, TimescaleDBPredictionsQueries, TimescaleDBResultsQueries
):
    """
    Handles all query operations for TimescaleDB connector.

    This class implements the query part of the TSDBConnector API with TimescaleDB-specific
    optimizations including pre-aggregate support for improved performance.

    The class uses multiple inheritance to compose functionality from domain-specific mixins:
    - TimescaleDBMetricsQueries: Metrics-related operations
    - TimescaleDBPredictionsQueries: Predictions and latency operations
    - TimescaleDBResultsQueries: Results and drift analysis operations

    All mixins share common utilities for query building, DataFrame processing, and
    pre-aggregate management to eliminate code duplication.
    """

    type: str = mm_schemas.TSDBTarget.TimescaleDB  # Assuming this exists in schemas
    schema = f"{timescaledb_schema._MODEL_MONITORING_SCHEMA}_{mlrun.mlconf.system_id}"

    def __init__(
        self,
        project: str,
        connection: TimescaleDBConnection,
        pre_aggregate_config: Optional[timescaledb_schema.PreAggregateConfig] = None,
    ):
        """
        Initialize the TimescaleDB query handler with all required attributes for mixins.

        :param project: Project name used for table naming and schema organization
        :param connection: TimescaleDB connection instance for executing queries
        :param pre_aggregate_config: Optional configuration for pre-aggregate optimizations
        """
        self.project = project
        self._connection = connection

        self._pre_aggregate_config = pre_aggregate_config
        self._pre_aggregate_handler = PreAggregateHandler(pre_aggregate_config)

        self._init_tables()

    def _init_tables(self):
        """Initialize the table schemas for TimescaleDB."""
        self.tables = {
            mm_schemas.TimescaleDBTables.APP_RESULTS: timescaledb_schema.AppResultTable(
                project=self.project, schema=self.schema
            ),
            mm_schemas.TimescaleDBTables.METRICS: timescaledb_schema.Metrics(
                project=self.project, schema=self.schema
            ),
            mm_schemas.TimescaleDBTables.PREDICTIONS: timescaledb_schema.Predictions(
                project=self.project, schema=self.schema
            ),
            mm_schemas.TimescaleDBTables.ERRORS: timescaledb_schema.Errors(
                project=self.project, schema=self.schema
            ),
        }

    def get_preaggregate_config(
        self,
    ) -> Optional[timescaledb_schema.PreAggregateConfig]:
        """Returns the current pre-aggregate configuration."""
        return self._pre_aggregate_config
