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
            agg_funcs=["max"] if use_pre_aggregates else None,
            interval=interval if use_pre_aggregates else None,
            order_by=table_schema.time_column,
            desc=True,
            use_pre_aggregates=use_pre_aggregates,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        if not df.empty:
            # Rename columns to match expected output
            column_mapping = {
                table_schema.time_column: mm_schemas.EventFieldType.LAST_REQUEST,
                mm_schemas.EventFieldType.LATENCY: "last_latency",
            }
            if use_pre_aggregates:
                column_mapping[f"max_{table_schema.time_column}"] = (
                    mm_schemas.EventFieldType.LAST_REQUEST
                )

            df.rename(columns=column_mapping, inplace=True)

            # Convert timestamp to proper format
            df[mm_schemas.EventFieldType.LAST_REQUEST] = pd.to_datetime(
                df[mm_schemas.EventFieldType.LAST_REQUEST],
                errors="coerce",
                utc=True,
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

        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        # Align times and check if we can use pre-aggregates
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.APP_RESULTS]
        filter_query = self._get_endpoint_filter(endpoint_ids)

        columns = [
            mm_schemas.ResultData.RESULT_STATUS,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            agg_funcs=["max"] if use_pre_aggregates else None,
            interval=interval if use_pre_aggregates else None,
            use_pre_aggregates=use_pre_aggregates,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        if not df.empty and use_pre_aggregates:
            df.rename(
                columns={
                    f"max_{mm_schemas.ResultData.RESULT_STATUS}": mm_schemas.ResultData.RESULT_STATUS
                },
                inplace=True,
            )

        return df

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

    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """Get error count with optional pre-aggregate optimization."""

        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        table_schema = self.tables[mm_schemas.TimescaleDBTables.ERRORS]
        filter_query = self._get_endpoint_filter(endpoint_ids)
        filter_query += f" AND {mm_schemas.EventFieldType.ERROR_TYPE} = '{mm_schemas.EventFieldType.INFER_ERROR}'"

        columns = [
            mm_schemas.EventFieldType.MODEL_ERROR,
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

        # Count errors by endpoint
        if not df.empty:
            df = (
                df.groupby(mm_schemas.WriterEvent.ENDPOINT_ID)
                .size()
                .reset_index(name="error_count")
            )

        return df

    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """Get average latency with optional pre-aggregate optimization."""

        start = start or (mlrun.utils.datetime_now() - timedelta(hours=24))
        start, end = self._pre_aggregate_handler.get_start_end(start, end)

        # Align times and check if we can use pre-aggregates
        start, end = self._pre_aggregate_handler.align_time_range(start, end, interval)
        use_pre_aggregates = self._pre_aggregate_handler.can_use_pre_aggregates(
            interval=interval
        )

        table_schema = self.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]
        filter_query = self._get_endpoint_filter(endpoint_ids)

        columns = [
            mm_schemas.EventFieldType.LATENCY,
            mm_schemas.WriterEvent.ENDPOINT_ID,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns,
            filter_query=filter_query,
            agg_funcs=["avg"] if use_pre_aggregates else None,
            interval=interval if use_pre_aggregates else None,
            use_pre_aggregates=use_pre_aggregates,
        )

        result = self._connection.run(query=query)
        df = pd.DataFrame(
            result.data if result else [], columns=result.fields if result else []
        )

        if not df.empty:
            if use_pre_aggregates:
                df.rename(
                    columns={f"avg_{mm_schemas.EventFieldType.LATENCY}": "avg_latency"},
                    inplace=True,
                )
            else:
                # Calculate average from raw data
                df = (
                    df.groupby(mm_schemas.WriterEvent.ENDPOINT_ID)[
                        mm_schemas.EventFieldType.LATENCY
                    ]
                    .mean()
                    .reset_index(name="avg_latency")
                )

        return df

    # In timescaledb_query_handler.py - Add this method to TimescaleDBQueryHandler

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

        # Build the aggregation query
        columns = [
            mm_schemas.WriterEvent.APPLICATION_NAME,
            mm_schemas.ResultData.RESULT_STATUS,
        ]

        query = table_schema._get_records_query(
            start=start,
            end=end,
            columns_to_filter=columns + ["COUNT(*) as count"],
            filter_query=filter_query,
            order_by=f"{mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.ResultData.RESULT_STATUS}",
        )

        # Modify query to add GROUP BY
        # This is a bit hacky, but TimescaleDB schema doesn't have built-in GROUP BY support
        if ";" in query:
            query = query.replace(
                ";",
                f" GROUP BY {mm_schemas.WriterEvent.APPLICATION_NAME}, {mm_schemas.ResultData.RESULT_STATUS};",
            )

        result = self._connection.run(query=query)

        if not result or not result.data:
            return {}

        return {(row[0], row[1]): row[2] for row in result.data}

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
            for metric in df_dictionary.keys():
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
