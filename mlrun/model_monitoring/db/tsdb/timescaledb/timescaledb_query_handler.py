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

from datetime import datetime, timedelta
from typing import Callable, Literal, Optional, Union

import pandas as pd
import v3io_frames.client

import mlrun
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.errors
import mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_schema as timescaledb_schema
import mlrun.utils
from mlrun.model_monitoring.db import TSDBConnector
from mlrun.model_monitoring.db.tsdb.preaggregate import (
    PreAggregateHandler,
)
from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_connection import (
    Statement,
    TimescaleDBConnection,
)
from mlrun.model_monitoring.helpers import get_invocations_fqn


class TimescaleDBQueryHandler:
    """
    Handles all query operations for TimescaleDB connector.

    This class implements the query part of the TSDBConnector API with TimescaleDB-specific
    optimizations including pre-aggregate support for improved performance.
    """

    type: str = mm_schemas.TSDBTarget.TimescaleDB  # Assuming this exists in schemas
    schema = f"{timescaledb_schema._MODEL_MONITORING_SCHEMA}_{mlrun.mlconf.system_id}"

    def __init__(
        self,
        project: str,
        connection: TimescaleDBConnection,
        pre_aggregate_config: Optional[timescaledb_schema.PreAggregateConfig] = None,
    ):
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

    def get_model_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str,
        end: str,
        interval: Optional[str] = None,
        agg_function: Optional[str] = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get real-time metrics with optional pre-aggregate optimization."""

        # Validate parameters using the pre-aggregate handler
        self._pre_aggregate_handler.validate_interval_and_function(
            interval, agg_function
        )

        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        # Align times if interval is provided
        start_dt, end_dt = self._pre_aggregate_handler.align_time_range(
            start_dt, end_dt, interval
        )

        # Check if we can use pre-aggregates
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval, agg_funcs=[agg_function] if agg_function else None
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Build query using the schema's query builder
        columns = [table_schema.time_column, mm_schemas.MetricData.METRIC_VALUE]
        filter_query = f"{mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_id}'"

        query = table_schema._get_records_query(
            start=start_dt,
            end=end_dt,
            columns_to_filter=columns,
            filter_query=filter_query,
            interval=interval if use_pre_aggregates else None,
            agg_funcs=[agg_function] if use_pre_aggregates and agg_function else None,
            use_pre_aggregates=use_pre_aggregates,
        )

        result = self._connection.run(query=query)

        # Process results into expected format
        metrics_data = {}
        if result and result.data:
            metric_name = (
                "default_metric"  # Would need to be extracted from actual query
            )
            for row in result.data:
                timestamp, value = row[0], row[1]
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append((timestamp.isoformat(), float(value)))

        return metrics_data

    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: Literal["metrics", "results"],
        with_result_extra_data: bool = False,
    ) -> Union[
        list[
            Union[
                mm_schemas.ModelEndpointMonitoringResultValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
        list[
            Union[
                mm_schemas.ModelEndpointMonitoringMetricValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ],
        ],
    ]:
        """Read metrics or results data from TimescaleDB."""

        if type == "metrics":
            table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]
            name_column = mm_schemas.MetricData.METRIC_NAME
            value_column = mm_schemas.MetricData.METRIC_VALUE
            columns = [
                table_schema.time_column,
                mm_schemas.WriterEvent.APPLICATION_NAME,
                name_column,
                value_column,
            ]
            df_handler = TSDBConnector.df_to_metrics_values  # Use base class method
        else:  # results
            table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
            name_column = mm_schemas.ResultData.RESULT_NAME
            value_column = mm_schemas.ResultData.RESULT_VALUE
            columns = [
                table_schema.time_column,
                mm_schemas.WriterEvent.APPLICATION_NAME,
                name_column,
                value_column,
                mm_schemas.ResultData.RESULT_STATUS,
                mm_schemas.ResultData.RESULT_KIND,
            ]
            if with_result_extra_data:
                columns.append(mm_schemas.ResultData.RESULT_EXTRA_DATA)
            df_handler = TSDBConnector.df_to_results_values  # Use base class method

        # Build metrics condition
        metrics_conditions = []
        for metric in metrics:
            condition = f"({mm_schemas.WriterEvent.APPLICATION_NAME}='{metric.app}' AND {name_column}='{metric.name}')"
            metrics_conditions.append(condition)

        metrics_condition = " OR ".join(metrics_conditions)
        filter_query = f"({mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_id}') AND ({metrics_condition})"

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
        )

        result = self._connection.run(query=query)

        # Convert to DataFrame
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        if not df.empty:
            df[table_schema.time_column] = pd.to_datetime(df[table_schema.time_column])
            df.set_index(table_schema.time_column, inplace=True)

        if not with_result_extra_data and type == "results":
            df[mm_schemas.ResultData.RESULT_EXTRA_DATA] = ""

        return df_handler(df=df, metrics=metrics, project=self.project)

    def read_predictions(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        aggregation_window: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        limit: Optional[int] = None,
        use_pre_aggregates: bool = True,
    ) -> Union[
        mm_schemas.ModelEndpointMonitoringMetricValues,
        mm_schemas.ModelEndpointMonitoringMetricNoData,
    ]:
        """Read predictions with optional pre-aggregate optimization."""

        if (agg_funcs and not aggregation_window) or (
            aggregation_window and not agg_funcs
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "both or neither of `aggregation_window` and `agg_funcs` must be provided"
            )

        # Align times if aggregation window is provided
        start, end = self._pre_aggregate_handler.align_time_range(
            start, end, aggregation_window
        )

        # Check if we can use pre-aggregates
        can_use_pre_aggregates = (
            use_pre_aggregates
            and self._pre_aggregate_handler.can_use_pre_aggregates(
                interval=aggregation_window, agg_funcs=agg_funcs
            )
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]

        filter_query = f"{mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_id}'"
        columns = [
            table_schema.time_column,
            mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            interval=aggregation_window if can_use_pre_aggregates else None,
            agg_funcs=agg_funcs if can_use_pre_aggregates else None,
            limit=limit,
            use_pre_aggregates=can_use_pre_aggregates,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        full_name = get_invocations_fqn(self.project)

        if df.empty:
            return mm_schemas.ModelEndpointMonitoringMetricNoData(
                full_name=full_name,
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )

        # Set up time index based on whether we used aggregation
        if aggregation_window and can_use_pre_aggregates:
            time_col = "time_bucket"
        else:
            time_col = table_schema.time_column

        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

        # Determine value column name
        if agg_funcs and can_use_pre_aggregates:
            value_col = (
                f"{agg_funcs[0]}_{mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT}"
            )
        else:
            value_col = mm_schemas.EventFieldType.ESTIMATED_PREDICTION_COUNT

        return mm_schemas.ModelEndpointMonitoringMetricValues(
            full_name=full_name,
            values=list(zip(df.index, df[value_col])),
        )

    def get_last_request(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get last request timestamp with optional pre-aggregate optimization."""

        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        # Align times and check if we can use pre-aggregates
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        filter_query = self._get_endpoint_filter(endpoint_ids)

        if use_pre_aggregates:
            columns = [
                mm_schemas.WriterEvent.ENDPOINT_ID,
                table_schema.time_column,
                mm_schemas.EventFieldType.LATENCY,
            ]

            query = table_schema._get_records_query(
                start=start,
                end=end,
                columns_to_filter=columns,
                filter_query=filter_query,
                agg_funcs=["max"],
                interval=interval,
                use_pre_aggregates=True,
            )

            result = self._connection.run(query=query)
            df = pd.DataFrame(
                result.data if result else [], columns=result.fields if result else []
            )

            if not df.empty:
                # Handle pre-aggregate column renaming
                column_mapping = {
                    f"max_{table_schema.time_column}": mm_schemas.EventFieldType.LAST_REQUEST,
                    f"max_{mm_schemas.EventFieldType.LATENCY}": "last_latency",
                }
                df.rename(columns=column_mapping, inplace=True)
                # Ensure consistent column naming
                df.rename(
                    columns={mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint_id"},
                    inplace=True,
                )
        else:
            # Use PostgreSQL DISTINCT ON for raw data - most efficient approach
            query = f"""
            SELECT DISTINCT ON ({mm_schemas.WriterEvent.ENDPOINT_ID})
                {mm_schemas.WriterEvent.ENDPOINT_ID} as endpoint_id,
                {table_schema.time_column} as {mm_schemas.EventFieldType.LAST_REQUEST},
                {mm_schemas.EventFieldType.LATENCY} as last_latency
            FROM {table_schema.schema}.{table_schema.table_name}
            WHERE {filter_query}
            AND {table_schema.time_column} >= '{start}'
            AND {table_schema.time_column} <= '{end}'
            ORDER BY {mm_schemas.WriterEvent.ENDPOINT_ID}, {table_schema.time_column} DESC;
            """

            result = self._connection.run(query=query)
            df = pd.DataFrame(
                result.data if result else [], columns=result.fields if result else []
            )

        # Convert timestamp to proper format (common for both paths)
        if not df.empty and mm_schemas.EventFieldType.LAST_REQUEST in df.columns:
            df[mm_schemas.EventFieldType.LAST_REQUEST] = pd.to_datetime(
                df[mm_schemas.EventFieldType.LAST_REQUEST],
                errors="coerce",
                utc=True,
            )

        return df

    #####
    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """Get average latency with optional pre-aggregate optimization."""

        # Convert single endpoint to list for consistent handling
        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        # Set default start time and get end time
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        # Align times and check if we can use pre-aggregates
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        filter_query = self._get_endpoint_filter(endpoint_ids)

        if use_pre_aggregates:
            try:
                # For pre-aggregates, don't specify columns - let the schema handle the continuous aggregate structure
                query = table_schema._get_records_query(
                    start=start,
                    end=end,
                    columns_to_filter=None,  # Don't specify columns for pre-aggregates
                    filter_query=filter_query,
                    agg_funcs=["avg"],
                    interval=interval,
                    use_pre_aggregates=True,
                )

                result = self._connection.run(query=query)
                df = pd.DataFrame(
                    result.data if result else [],
                    columns=result.fields if result else [],
                )

                if not df.empty:
                    # Handle flexible column naming - find columns that contain the expected data
                    column_mapping = {}

                    # Look for latency-related column (avg_latency, avg_xxx, etc.)
                    latency_col = None
                    for col in df.columns:
                        if "avg" in col.lower() and (
                            "latency" in col.lower() or col.endswith("latency")
                        ):
                            latency_col = col
                            break
                    if latency_col and latency_col != "avg_latency":
                        column_mapping[latency_col] = "avg_latency"

                    # Look for endpoint-related column
                    endpoint_col = None
                    for col in df.columns:
                        if "endpoint" in col.lower():
                            endpoint_col = col
                            break
                    if endpoint_col and endpoint_col != "endpoint_id":
                        column_mapping[endpoint_col] = "endpoint_id"

                    if column_mapping:
                        df.rename(columns=column_mapping, inplace=True)

                    return df

            except Exception as e:
                # If pre-aggregate query fails, fall back to raw data
                print(f"Pre-aggregate query failed, falling back to raw data: {e}")
                use_pre_aggregates = False

        # Use the schema's _get_records_query method for raw data aggregation
        columns = [
            f"{mm_schemas.WriterEvent.ENDPOINT_ID} as endpoint_id",
            f"AVG({mm_schemas.EventFieldType.LATENCY}) as avg_latency",
        ]

        group_by_columns = [mm_schemas.WriterEvent.ENDPOINT_ID]

        # Add additional filter to exclude invalid latency values
        enhanced_filter_query = filter_query
        if enhanced_filter_query:
            enhanced_filter_query += (
                f" AND {mm_schemas.EventFieldType.LATENCY} IS NOT NULL"
            )
        else:
            enhanced_filter_query = f"{mm_schemas.EventFieldType.LATENCY} IS NOT NULL"
        enhanced_filter_query += f" AND {mm_schemas.EventFieldType.LATENCY} > 0"

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=enhanced_filter_query,
            group_by=group_by_columns,
            order_by=mm_schemas.WriterEvent.ENDPOINT_ID,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        return df

    def get_drift_status(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """Get drift status with optional pre-aggregate optimization."""

        # Convert single endpoint to list for consistent handling
        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        # Set default start time and get end time
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        # Align times and check if we can use pre-aggregates
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        filter_query = self._get_endpoint_filter(endpoint_ids)

        if use_pre_aggregates:
            try:
                # For pre-aggregates, don't specify columns - let the schema handle the continuous aggregate structure
                query = table_schema._get_records_query(
                    start=start,
                    end=end,
                    columns_to_filter=None,  # Don't specify columns for pre-aggregates
                    filter_query=filter_query,
                    agg_funcs=["max"],
                    interval=interval,
                    use_pre_aggregates=True,
                )

                result = self._connection.run(query=query)
                df = pd.DataFrame(
                    result.data if result else [],
                    columns=result.fields if result else [],
                )

                if not df.empty:
                    # Handle flexible column naming - find columns that contain the expected data
                    column_mapping = {}

                    # Look for status-related column (max_result_status, max_xxx, etc.)
                    status_col = None
                    for col in df.columns:
                        if "max" in col.lower() and (
                            "status" in col.lower() or "result" in col.lower()
                        ):
                            status_col = col
                            break
                        elif col == mm_schemas.ResultData.RESULT_STATUS:
                            status_col = col
                            break
                    if status_col and status_col != mm_schemas.ResultData.RESULT_STATUS:
                        column_mapping[status_col] = mm_schemas.ResultData.RESULT_STATUS

                    # Look for endpoint-related column
                    endpoint_col = None
                    for col in df.columns:
                        if "endpoint" in col.lower():
                            endpoint_col = col
                            break
                    if endpoint_col and endpoint_col != "endpoint_id":
                        column_mapping[endpoint_col] = "endpoint_id"

                    if column_mapping:
                        df.rename(columns=column_mapping, inplace=True)

                    return df

            except Exception as e:
                # If pre-aggregate query fails, fall back to raw data
                print(f"Pre-aggregate query failed, falling back to raw data: {e}")
                use_pre_aggregates = False

        # Use the schema's _get_records_query method for raw data aggregation
        columns = [
            f"{mm_schemas.WriterEvent.ENDPOINT_ID} as endpoint_id",
            f"MAX({mm_schemas.ResultData.RESULT_STATUS}) as {mm_schemas.ResultData.RESULT_STATUS}",
        ]

        group_by_columns = [mm_schemas.WriterEvent.ENDPOINT_ID]

        # Add filter to exclude NULL result status values
        enhanced_filter_query = filter_query
        if enhanced_filter_query:
            enhanced_filter_query += (
                f" AND {mm_schemas.ResultData.RESULT_STATUS} IS NOT NULL"
            )
        else:
            enhanced_filter_query = f"{mm_schemas.ResultData.RESULT_STATUS} IS NOT NULL"

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=enhanced_filter_query,
            group_by=group_by_columns,
            order_by=mm_schemas.WriterEvent.ENDPOINT_ID,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        return df

    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get error count with optional pre-aggregate optimization."""

        # Convert single endpoint to list for consistent handling
        if isinstance(endpoint_ids, str):
            endpoint_ids = [endpoint_ids]

        # Set default start time and get end time
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        # Align times and check if we can use pre-aggregates
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.ERRORS]
        filter_query = self._get_endpoint_filter(endpoint_ids)

        # The error_type filter might not be available in continuous aggregates
        # so we'll try with pre-aggregates first, then fall back to raw data with the filter

        if use_pre_aggregates:
            try:
                # Try pre-aggregates WITHOUT the error_type filter first
                # If the continuous aggregate was created with the filter, this will work
                # If not, we'll fall back to raw data
                query = table_schema._get_records_query(
                    start=start,
                    end=end,
                    columns_to_filter=None,  # Don't specify columns for pre-aggregates
                    filter_query=filter_query,  # Only endpoint filter, no error_type
                    agg_funcs=["count"],
                    interval=interval,
                    use_pre_aggregates=True,
                )

                result = self._connection.run(query=query)
                df = pd.DataFrame(
                    result.data if result else [],
                    columns=result.fields if result else [],
                )

                if not df.empty:
                    # Handle flexible column naming - find columns that contain the expected data
                    column_mapping = {}

                    # Look for count-related column (count_model_error, count_xxx, count, etc.)
                    count_col = None
                    for col in df.columns:
                        if "count" in col.lower():
                            count_col = col
                            break
                    if count_col and count_col != "error_count":
                        column_mapping[count_col] = "error_count"

                    # Look for endpoint-related column
                    endpoint_col = None
                    for col in df.columns:
                        if "endpoint" in col.lower():
                            endpoint_col = col
                            break
                    if endpoint_col and endpoint_col != "endpoint_id":
                        column_mapping[endpoint_col] = "endpoint_id"

                    if column_mapping:
                        df.rename(columns=column_mapping, inplace=True)

                    return df

            except Exception as e:
                # If pre-aggregate query fails (likely due to missing error_type column),
                # fall back to raw data with full filtering
                print(
                    f"Pre-aggregate query failed, falling back to raw data with error_type filter: {e}"
                )
                use_pre_aggregates = False

        # Use PostgreSQL aggregation with GROUP BY for raw data WITH error_type filter
        filter_query += f" AND {mm_schemas.EventFieldType.ERROR_TYPE} = '{mm_schemas.EventFieldType.INFER_ERROR}'"

        query = f"""
        SELECT
            {mm_schemas.WriterEvent.ENDPOINT_ID} as endpoint_id,
            COUNT(*) as error_count
        FROM {table_schema.schema}.{table_schema.table_name}
        WHERE {filter_query}
        AND {table_schema.time_column} >= '{start}'
        AND {table_schema.time_column} <= '{end}'
        GROUP BY {mm_schemas.WriterEvent.ENDPOINT_ID};
        """

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        return df

    #####
    def get_metrics_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get metrics metadata with optional pre-aggregate optimization."""

        start, end = self._pre_aggregate_handler.get_start_end(start, end)
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]
        filter_query = self._get_endpoint_filter(endpoint_id)

        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.MetricData.METRIC_NAME,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        # Get distinct values
        if not df.empty:
            df = df.drop_duplicates()

        return df

    def get_results_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get results metadata with optional pre-aggregate optimization."""

        start, end = self._pre_aggregate_handler.get_start_end(start, end)
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        filter_query = self._get_endpoint_filter(endpoint_id)

        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_NAME,
            mm_schemas.ResultData.RESULT_KIND,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        # Get distinct values
        if not df.empty:
            df = df.drop_duplicates()

        return df

    def count_results_by_status(
        self,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        endpoint_ids: Optional[Union[str, list[str]]] = None,
        application_names: Optional[Union[str, list[str]]] = None,
        result_status_list: Optional[list[int]] = None,
    ) -> dict[tuple[str, int], int]:
        """
        Read results status from the TSDB and return a dictionary of results statuses by application name.

        :param start:              The start time in which to read the results. By default, the last 24 hours are read.
        :param end:                The end time in which to read the results. Default is the current time (now).
        :param endpoint_ids:       Optional list of endpoint ids to filter the results by. By default, all
                                endpoint ids are included.
        :param application_names:  Optional list of application names to filter the results by. By default, all
                                application are included.
        :param result_status_list: Optional list of result statuses to filter the results by. By default, all
                                result statuses are included.

        :return: A dictionary where the key is a tuple of (application_name, result_status) and the value is the total
                number of results with that status for that application.
                For example:
                {
                    ('app1', 1): 10,
                    ('app1', 2): 5
                }
        """
        # Set defaults
        now = mlrun.utils.datetime_now()
        start = start or (now - timedelta(hours=24))
        end = end or now

        # Convert string dates to datetime if needed
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        # Build filter conditions
        filter_conditions = []

        if endpoint_ids:
            endpoint_filter = self._get_endpoint_filter(endpoint_ids)
            filter_conditions.append(endpoint_filter)

        if application_names:
            if isinstance(application_names, str):
                app_filter = (
                    f"{mm_schemas.WriterEvent.APPLICATION_NAME} = '{application_names}'"
                )
            else:
                app_list = "', '".join(application_names)
                app_filter = (
                    f"{mm_schemas.WriterEvent.APPLICATION_NAME} IN ('{app_list}')"
                )
            filter_conditions.append(app_filter)

        if result_status_list:
            if len(result_status_list) == 1:
                status_filter = (
                    f"{mm_schemas.ResultData.RESULT_STATUS} = {result_status_list[0]}"
                )
            else:
                status_list = ", ".join(map(str, result_status_list))
                status_filter = (
                    f"{mm_schemas.ResultData.RESULT_STATUS} IN ({status_list})"
                )
            filter_conditions.append(status_filter)

        filter_query = " AND ".join(filter_conditions) if filter_conditions else None

        # Build the aggregation query using the enhanced _get_records_query
        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_STATUS,
            "COUNT(*) as count",
        ]

        group_by_columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_STATUS,
        ]

        order_by_clause = f"{mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.ResultData.RESULT_STATUS}"

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            group_by=group_by_columns,
            order_by=order_by_clause,
        )

        result = self._connection.run(query=query)

        if not result or not result.data:
            return {}

        return {(row[0], row[1]): row[2] for row in result.data}

    def count_processed_model_endpoints(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        application_names: Optional[Union[str, list[str]]] = None,
    ) -> dict[str, int]:
        """
        Optimized count with application filtering using JOIN approach.

        This implementation:
        1. Uses JOIN when application filtering is needed (most performant)
        2. Falls back to simple query when no filtering (fastest for that case)
        3. Leverages TimescaleDB's chunk exclusion and parallel processing
        4. Can utilize pre-aggregates when available
        """
        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        predictions_table = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]

        if application_names:
            # Ensure application_names is a list
            if isinstance(application_names, str):
                application_names = [application_names]

            result = {}

            # For each application, call the existing JOIN method and wrap result in dict
            for app_name in application_names:
                # Use existing _count_with_application_join but extract count for single app
                count = self._count_with_application_join(
                    predictions_table,
                    start,
                    end,
                    [app_name],  # Pass as list to existing method
                )
                result[app_name] = count

            return result
        else:
            # Use existing simple count method and wrap result
            total_count = self._count_simple(predictions_table, start, end)
            return {"total": total_count} if total_count > 0 else {}

    def _count_with_application_join(
        self,
        predictions_table,
        start: datetime,
        end: datetime,
        application_names: Union[str, list[str]],
    ) -> int:
        """
        Use JOIN with metrics table for application filtering.

        Performance characteristics:
        - Leverages indexes on both tables
        - TimescaleDB optimizes time-based JOINs
        - Chunk exclusion works on both sides
        - DISTINCT applied after filtering
        """
        metrics_table = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Normalize application_names to list for consistent handling
        if isinstance(application_names, str):
            app_names_list = [application_names]
        else:
            app_names_list = list(application_names)

        # Build parameterized query with proper placeholders
        app_placeholders = ", ".join(["%s"] * len(app_names_list))

        query_sql = f"""
        SELECT COUNT(DISTINCT p.{mm_schemas.WriterEvent.ENDPOINT_ID}) as endpoint_count
        FROM {predictions_table.schema}.{predictions_table.table_name} p
        INNER JOIN {metrics_table.schema}.{metrics_table.table_name} m
            ON p.{mm_schemas.WriterEvent.ENDPOINT_ID} = m.{mm_schemas.WriterEvent.ENDPOINT_ID}
            AND m.{metrics_table.time_column} >= %s
            AND m.{metrics_table.time_column} <= %s
        WHERE p.{predictions_table.time_column} >= %s
            AND p.{predictions_table.time_column} <= %s
            AND m.{mm_schemas.WriterEvent.APPLICATION_NAME} IN ({app_placeholders})
        """

        # Parameters: [start, end, start, end] + application_names_list
        params = [start, end, start, end] + app_names_list
        stmt = Statement(query_sql, params)
        result = self._connection.run(query=stmt)

        return result.data[0][0] if result and result.data else 0

    def _count_simple(self, predictions_table, start: datetime, end: datetime) -> int:
        """
        Simple count without application filtering.

        Uses the schema's query builder for consistency and potential pre-aggregate usage.
        """
        columns = [
            f"COUNT(DISTINCT {mm_schemas.WriterEvent.ENDPOINT_ID}) as endpoint_count"
        ]

        query = predictions_table._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
        )

        result = self._connection.run(query=query)
        return result.data[0][0] if result and result.data else 0

    def calculate_latest_metrics(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        application_names: Optional[list[str]] = None,
    ) -> list[
        Union[mm_schemas.ApplicationResultRecord, mm_schemas.ApplicationMetricRecord]
    ]:
        """
        Calculate the latest metrics and results across applications.

        Returns a list of ApplicationResultRecord and ApplicationMetricRecord objects.

        :param start:              The start time of the query. Last 24 hours is used by default.
        :param end:                The end time of the query. The current time is used by default.
        :param application_names:  A list of application names to filter the results by. If not provided, all
                                applications are included.
        :return:                   A list containing the latest metrics and results for each application.
        """
        if not application_names:
            return []

        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        metric_objects = []

        for app_name in application_names:
            # Get latest results for this application
            results_df = self._get_latest_results_for_application(app_name, start, end)
            if not results_df.empty:
                for _, row in results_df.iterrows():
                    metric_objects.append(
                        mm_schemas.ApplicationResultRecord(
                            time=row.get("end_infer_time"),
                            value=row.get("result_value"),
                            kind=mm_schemas.ResultKindApp(row.get("result_kind")),
                            status=mm_schemas.ResultStatusApp(row.get("result_status")),
                            result_name=row.get("result_name"),
                        )
                    )

            # Get latest metrics for this application
            metrics_df = self._get_latest_metrics_for_application(app_name, start, end)
            if not metrics_df.empty:
                for _, row in metrics_df.iterrows():
                    metric_objects.append(
                        mm_schemas.ApplicationMetricRecord(
                            time=row.get("end_infer_time"),
                            value=row.get("metric_value"),
                            metric_name=row.get("metric_name"),
                        )
                    )

        return metric_objects

    def _get_latest_results_for_application(
        self, application_name: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get the latest results for a specific application."""
        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        # Get the latest results grouped by result_name
        query = f"""
        SELECT DISTINCT ON (result_name)
            {mm_schemas.WriterEvent.END_INFER_TIME},
            {mm_schemas.WriterEvent.APPLICATION_NAME},
            {mm_schemas.ResultData.RESULT_NAME},
            {mm_schemas.ResultData.RESULT_VALUE},
            {mm_schemas.ResultData.RESULT_STATUS},
            {mm_schemas.ResultData.RESULT_KIND}
        FROM {table_schema.schema}.{table_schema.table_name}
        WHERE {mm_schemas.WriterEvent.APPLICATION_NAME} = %s
        AND {mm_schemas.WriterEvent.END_INFER_TIME} >= %s
        AND {mm_schemas.WriterEvent.END_INFER_TIME} <= %s
        ORDER BY result_name, {mm_schemas.WriterEvent.END_INFER_TIME} DESC
        """

        stmt = Statement(query, (application_name, start, end))
        result = self._connection.run(query=stmt)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        return df

    def _get_latest_metrics_for_application(
        self, application_name: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Get the latest metrics for a specific application."""
        table_schema = self.tables[mm_schemas.TimescaleDBTables.METRICS]

        # Get the latest metrics grouped by metric_name
        query = f"""
        SELECT DISTINCT ON (metric_name)
            {mm_schemas.WriterEvent.END_INFER_TIME},
            {mm_schemas.WriterEvent.APPLICATION_NAME},
            {mm_schemas.MetricData.METRIC_NAME},
            {mm_schemas.MetricData.METRIC_VALUE}
        FROM {table_schema.schema}.{table_schema.table_name}
        WHERE {mm_schemas.WriterEvent.APPLICATION_NAME} = %s
        AND {mm_schemas.WriterEvent.END_INFER_TIME} >= %s
        AND {mm_schemas.WriterEvent.END_INFER_TIME} <= %s
        ORDER BY metric_name, {mm_schemas.WriterEvent.END_INFER_TIME} DESC
        """

        stmt = Statement(query, (application_name, start, end))
        result = self._connection.run(query=stmt)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )
        return df

    def get_drift_data(
        self,
        start: datetime,
        end: datetime,
        interval: Optional[str] = None,
    ) -> mm_schemas.ModelEndpointDriftValues:
        """
        Get drift data aggregated by time intervals, showing the count of suspected and detected drift events.

        This method queries the app_results table for drift-related statuses (potential_detection=1, detected=2)
        and aggregates them by time intervals, counting the maximum drift status per endpoint per interval.

        :param start: Start time for the query
        :param end: End time for the query
        :param interval: Optional time interval for aggregation (e.g., "1 hour", "30 minutes").
                        If not provided, will be automatically determined based on query duration.
        :return: ModelEndpointDriftValues containing time-binned drift counts
        """
        # Align start/end times and determine interval
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        if interval is None:
            # Use automatic interval selection if not specified
            start, end, interval = self._prepare_aligned_start_end(start, end)
        else:
            # Use provided interval, align times to interval boundaries for consistency
            start, end = self._pre_aggregate_handler.align_time_range(
                start, end, interval
            )

        # Build status filter for drift-related statuses only
        suspected_status = mm_schemas.ResultStatusApp.potential_detection.value  # 1
        detected_status = mm_schemas.ResultStatusApp.detected.value  # 2

        app_results_table = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]

        # Use TimescaleDB's time_bucket function for interval aggregation
        # This is equivalent to TDEngine's INTERVAL() with PARTITION BY
        query = f"""
        WITH drift_intervals AS (
            SELECT
                time_bucket('{interval}', {mm_schemas.WriterEvent.END_INFER_TIME}) AS bucket_start,
                {mm_schemas.WriterEvent.ENDPOINT_ID},
                MAX({mm_schemas.ResultData.RESULT_STATUS}) AS max_status
            FROM {app_results_table.schema}.{app_results_table.table_name}
            WHERE {mm_schemas.ResultData.RESULT_STATUS} IN (%s, %s)
            AND {mm_schemas.WriterEvent.END_INFER_TIME} >= %s
            AND {mm_schemas.WriterEvent.END_INFER_TIME} <= %s
            GROUP BY bucket_start, {mm_schemas.WriterEvent.ENDPOINT_ID}
        )
        SELECT
            bucket_start,
            max_status,
            COUNT(*) as status_count
        FROM drift_intervals
        GROUP BY bucket_start, max_status
        ORDER BY bucket_start, max_status
        """

        # Execute the query with parameterized values
        stmt = Statement(query, (suspected_status, detected_status, start, end))
        result = self._connection.run(query=stmt)

        if not result or not result.data:
            return mm_schemas.ModelEndpointDriftValues(values=[])

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(result.data, columns=result.fields)

        # Use the shared _df_to_drift_data method to convert to the expected format
        return self._df_to_drift_data(df)

    def _prepare_aligned_start_end(
        self, start: datetime, end: datetime
    ) -> tuple[datetime, datetime, str]:
        """
        Prepare aligned start and end times with appropriate interval for drift data aggregation.

        This matches TDEngine's behavior of aligning times to hour boundaries.

        :param start: Original start time
        :param end: Original end time
        :return: Tuple of (aligned_start, aligned_end, interval_string)
        """
        # Default to 1 hour intervals like TDEngine
        interval = "1 hour"

        # Align to hour boundaries
        aligned_start = start.replace(minute=0, second=0, microsecond=0)

        # Calculate the time difference to determine appropriate interval
        time_diff = end - start

        if time_diff <= timedelta(hours=6):
            # For short periods, use 1 hour intervals
            interval = "1 hour"
        elif time_diff <= timedelta(days=2):
            # For medium periods, use 1 hour intervals
            interval = "1 hour"
        elif time_diff <= timedelta(days=7):
            # For week-long periods, use 6 hour intervals
            interval = "6 hours"
        else:
            # For longer periods, use daily intervals
            interval = "1 day"
            aligned_start = aligned_start.replace(hour=0)

        return aligned_start, end, interval

    def _df_to_drift_data(
        self, df: pd.DataFrame
    ) -> mm_schemas.ModelEndpointDriftValues:
        """
        Convert DataFrame with drift data to ModelEndpointDriftValues format.

        Expected DataFrame columns:
        - bucket_start: timestamp of the interval bucket
        - max_status: the maximum drift status in that bucket (1=suspected, 2=detected)
        - status_count: count of endpoints with that status in the bucket

        :param df: DataFrame with aggregated drift data
        :return: ModelEndpointDriftValues with time-binned counts
        """
        if df.empty:
            return mm_schemas.ModelEndpointDriftValues(values=[])

        suspected_val = mm_schemas.ResultStatusApp.potential_detection.value  # 1
        detected_val = mm_schemas.ResultStatusApp.detected.value  # 2

        # Rename columns to match the expected format from TDEngine
        df = df.rename(
            columns={
                "bucket_start": "_wstart",
                "max_status": f"max({mm_schemas.ResultData.RESULT_STATUS})",
                "status_count": "count",
            }
        )

        # Pivot the data to have separate columns for suspected and detected counts
        aggregated_df = (
            df.groupby(["_wstart", f"max({mm_schemas.ResultData.RESULT_STATUS})"])[
                "count"
            ]
            .sum()  # Sum counts for each interval x result-status combination
            .unstack()  # Create separate columns for each result-status
            .reindex(
                columns=[suspected_val, detected_val], fill_value=0
            )  # Ensure both columns exist
            .fillna(0)
            .astype(int)
            .rename(
                columns={
                    suspected_val: "count_suspected",
                    detected_val: "count_detected",
                }
            )
        )

        # Convert to list of tuples: (timestamp, count_suspected, count_detected)
        values = list(
            zip(
                aggregated_df.index,
                aggregated_df["count_suspected"],
                aggregated_df["count_detected"],
            )
        )

        return mm_schemas.ModelEndpointDriftValues(values=values)

    async def add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        project: str,
        run_in_threadpool: Callable,
        metric_list: Optional[list[str]] = None,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        """
        Add basic metrics to the model endpoint object using TimescaleDB optimizations.

        :param model_endpoint_objects: A list of `ModelEndpoint` objects that will
                                        be filled with the relevant basic metrics.
        :param project:                The name of the project.
        :param run_in_threadpool:      A function that runs another function in a thread pool.
        :param metric_list:            List of metrics to include from the time series DB. Defaults to all metrics.

        :return: A list of `ModelEndpointMonitoringMetric` objects.
        """

        uids = [mep.metadata.uid for mep in model_endpoint_objects]

        metric_name_to_function = {
            "error_count": self.get_error_count,
            "last_request": self.get_last_request,
            "avg_latency": self.get_avg_latency,
            "result_status": self.get_drift_status,
        }
        if metric_list is not None:
            for metric_name in list(metric_name_to_function):
                if metric_name not in metric_list:
                    del metric_name_to_function[metric_name]

        metric_name_to_df = {
            metric_name: function(endpoint_ids=uids)
            for metric_name, function in metric_name_to_function.items()
        }

        def add_metrics(
            mep: mlrun.common.schemas.ModelEndpoint,
            df_dictionary: dict[str, pd.DataFrame],
        ):
            for metric in df_dictionary:
                df = df_dictionary.get(metric, pd.DataFrame())
                if not df.empty:
                    line = df[df["endpoint_id"] == mep.metadata.uid]
                    if not line.empty and metric in line:
                        value = line[metric].item()
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        setattr(mep.status, metric, value)

            return mep

        return list(
            map(
                lambda mep: add_metrics(
                    mep=mep,
                    df_dictionary=metric_name_to_df,
                ),
                model_endpoint_objects,
            )
        )

    # Helper methods - using static methods from base class where possible
    @staticmethod
    def _get_endpoint_filter(endpoint_id: Union[str, list[str]]) -> str:
        """Generate SQL filter for endpoint IDs."""
        if isinstance(endpoint_id, str):
            return f"{mm_schemas.WriterEvent.ENDPOINT_ID}='{endpoint_id}'"
        elif isinstance(endpoint_id, list):
            endpoint_list = "', '".join(endpoint_id)
            return f"{mm_schemas.WriterEvent.ENDPOINT_ID} IN ('{endpoint_list}')"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid 'endpoint_id' filter: must be a string or a list."
            )
