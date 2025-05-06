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

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, ClassVar, Literal, Optional, Union

import pandas as pd
import pydantic.v1
import v3io_frames.client

import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.model_monitoring.db.tsdb.helpers
import mlrun.model_monitoring.helpers
from mlrun.utils import logger


class TSDBConnector(ABC):
    type: ClassVar[str]

    def __init__(self, project: str) -> None:
        """
        Initialize a new TSDB connector. The connector is used to interact with the TSDB and store monitoring data.
        At the moment we have 3 different types of monitoring data:
        - real time performance metrics: real time performance metrics that are being calculated by the model
        monitoring stream pod.
        Among these metrics are the base metrics (average latency and predictions over time), endpoint features
        (data samples), and custom metrics (user-defined metrics).
        - app_results: a detailed results that include status, kind, extra data, etc. These results are being calculated
        through the monitoring applications and stored in the TSDB using the model monitoring writer.
        - metrics: a basic key value that represents a numeric metric. Similar to the app_results, these metrics
        are being calculated through the monitoring applications and stored in the TSDB using the model monitoring
        writer.

        :param project: the name of the project.
        """
        self.project = project

    @abstractmethod
    def apply_monitoring_stream_steps(self, graph, **kwargs) -> None:
        """
        Apply TSDB steps on the provided monitoring graph. Throughout these steps, the graph stores live data of
        different key metric dictionaries. This data is being used by the monitoring dashboards in
        grafana.
        There are 3 different key metric dictionaries that are being generated throughout these steps:
        - base_metrics (average latency and predictions over time)
        - endpoint_features (Prediction and feature names and values)
        - custom_metrics (user-defined metrics)
        """
        pass

    @abstractmethod
    def handle_model_error(self, graph, **kwargs) -> None:
        """
        Adds a branch to the stream pod graph to handle events that
        arrive with errors from the model server and saves them to the error TSDB table.
        The first step that generates by this method should come after `ForwardError` step.
        """

    @abstractmethod
    def write_application_event(
        self,
        event: dict,
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a single application or metric to TSDB.

        :raise mlrun.errors.MLRunRuntimeError: If an error occurred while writing the event.
        """

    @abstractmethod
    def delete_tsdb_records(
        self,
        endpoint_ids: list[str],
    ) -> None:
        """
        Delete model endpoint records from the TSDB connector.
        :param endpoint_ids: List of model endpoint unique identifiers.
        :param delete_timeout: The timeout in seconds to wait for the deletion to complete.
        """
        pass

    @abstractmethod
    def delete_tsdb_resources(self):
        """
        Delete all project resources in the TSDB connector, such as model endpoints data and drift results.
        """
        pass

    @abstractmethod
    def get_model_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: list[str],
        start: str,
        end: str,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Getting real time metrics from the TSDB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user. Note that these
        metrics are being calculated by the model monitoring stream pod.
        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of real-time metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an  RFC 3339
                                 time, a  Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and `'s'`
                                 = seconds), or 0 for the earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an  RFC 3339
                                 time, a  Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and `'s'`
                                 = seconds), or 0 for the earliest time.
        :return: A dictionary of metrics in which the key is a metric name and the value is a list of tuples that
                 includes timestamps and the values.
        """
        pass

    @abstractmethod
    def create_tables(self) -> None:
        """
        Create the TSDB tables using the TSDB connector. At the moment we support 3 types of tables:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a numeric metric.
        - predictions: latency of each prediction.
        """

    @abstractmethod
    def read_metrics_data(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        type: Literal["metrics", "results"],
        with_result_extra_data: bool,
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
        """
        Read metrics OR results from the TSDB and return as a list.

        :param endpoint_id: The model endpoint identifier.
        :param start:                  The start time of the query.
        :param end:                    The end time of the query.
        :param metrics:                The list of metrics to get the values for.
        :param type:                   "metrics" or "results" - the type of each item in metrics.
        :param with_result_extra_data: Whether to include the extra data in the results, relevant only when
                                       `type="results"`.
        :return:                        A list of result values or a list of metric values.
        """

    @abstractmethod
    def read_predictions(
        self,
        *,
        endpoint_id: str,
        start: datetime,
        end: datetime,
        aggregation_window: Optional[str] = None,
        agg_funcs: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> Union[
        mm_schemas.ModelEndpointMonitoringMetricValues,
        mm_schemas.ModelEndpointMonitoringMetricNoData,
    ]:
        """
        Read the "invocations" metric for the provided model endpoint in the given time range,
        and return the metric values if any, otherwise signify with the "no data" object.

        :param endpoint_id:        The model endpoint identifier.
        :param start:              The start time of the query.
        :param end:                The end time of the query.
        :param aggregation_window: On what time window length should the invocations be aggregated. If provided,
                                   the `agg_funcs` must be provided as well. Provided as a string in the format of '1m',
                                   '1h', etc.
        :param agg_funcs:          List of aggregation functions to apply on the invocations. If provided, the
                                   `aggregation_window` must be provided as well. Provided as a list of strings in
                                   the format of ['sum', 'avg', 'count', ...]
        :param limit:              The maximum number of records to return.

        :raise mlrun.errors.MLRunInvalidArgumentError: If only one of `aggregation_window` and `agg_funcs` is provided.
        :return:                   Metric values object or no data object.
        """

    @abstractmethod
    def get_last_request(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Union[pd.DataFrame, dict[str, float]]:
        """
        Fetches data from the predictions TSDB table and returns the most recent request
        timestamp for each specified endpoint.

        :param endpoint_ids:    A list of model endpoint identifiers.
        :param start:           The start time for the query.
        :param end:             The end time for the query.

        :return: A pd.DataFrame containing the columns [endpoint_id, last_request, last_latency] or a dictionary
        containing the endpoint_id as the key and the last request timestamp as the value.
        if an endpoint has not been invoked within the specified time range, it will not appear in the result (relevant
        only to non-v3io connector).
        """

    @abstractmethod
    def get_drift_status(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """
        Fetches data from the app-results TSDB table and returns the highest status among all
        the result in the provided time range, which by default is the last 24 hours, for each specified endpoint.

        :param endpoint_ids:    A list of model endpoint identifiers.
        :param start:           The start time for the query.
        :param end:             The end time for the query.
        :param get_raw:         Whether to return the request as raw frames rather than a pandas dataframe. Defaults
          to False. This can greatly improve performance when a dataframe isn't needed.

        :return: A pd.DataFrame containing the columns [result_status, endpoint_id].
        If an endpoint has not been monitored within the specified time range (last 24 hours),
        it will not appear in the result.
        """

    @abstractmethod
    def get_metrics_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetches distinct metrics metadata from the metrics TSDB table for a specified model endpoints.

        :param endpoint_id:        The model endpoint identifier. Can be a single id or a list of ids.
        :param start:              The start time of the query.
        :param end:                The end time of the query.

        :return: A pd.DataFrame containing all distinct metrics for the specified endpoint within the given time range.
        Containing the columns [application_name, metric_name, endpoint_id]
        """

    @abstractmethod
    def get_results_metadata(
        self,
        endpoint_id: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetches distinct results metadata from the app-results TSDB table for a specified model endpoints.

        :param endpoint_id:        The model endpoint identifier. Can be a single id or a list of ids.
        :param start:              The start time of the query.
        :param end:                The end time of the query.

        :return: A pd.DataFrame containing all distinct results for the specified endpoint within the given time range.
        Containing the columns [application_name, result_name, result_kind, endpoint_id]
        """

    @abstractmethod
    def get_error_count(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """
        Fetches data from the error TSDB table and returns the error count for each specified endpoint.

        :param endpoint_ids:    A list of model endpoint identifiers.
        :param start:           The start time for the query.
        :param end:             The end time for the query.
        :param get_raw:         Whether to return the request as raw frames rather than a pandas dataframe. Defaults
          to False. This can greatly improve performance when a dataframe isn't needed.

        :return: A pd.DataFrame containing the columns [error_count, endpoint_id].
        If an endpoint have not raised error within the specified time range, it will not appear in the result.
        """

    @abstractmethod
    def get_avg_latency(
        self,
        endpoint_ids: Union[str, list[str]],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        get_raw: bool = False,
    ) -> Union[pd.DataFrame, list[v3io_frames.client.RawFrame]]:
        """
        Fetches data from the predictions TSDB table and returns the average latency for each specified endpoint
        in the provided time range, which by default is the last 24 hours.

        :param endpoint_ids:    A list of model endpoint identifiers.
        :param start:           The start time for the query.
        :param end:             The end time for the query.
        :param get_raw:         Whether to return the request as raw frames rather than a pandas dataframe. Defaults
          to False. This can greatly improve performance when a dataframe isn't needed.

        :return: A pd.DataFrame containing the columns [avg_latency, endpoint_id].
        If an endpoint has not been invoked within the specified time range, it will not appear in the result.
        """

    async def add_basic_metrics(
        self,
        model_endpoint_objects: list[mlrun.common.schemas.ModelEndpoint],
        project: str,
        run_in_threadpool: Callable,
        metric_list: Optional[list[str]] = None,
    ) -> list[mlrun.common.schemas.ModelEndpoint]:
        raise NotImplementedError()

    @staticmethod
    def df_to_metrics_values(
        *,
        df: pd.DataFrame,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        project: str,
    ) -> list[
        Union[
            mm_schemas.ModelEndpointMonitoringMetricValues,
            mm_schemas.ModelEndpointMonitoringMetricNoData,
        ]
    ]:
        """
        Parse a time-indexed DataFrame of metrics from the TSDB into a list of
        metrics values per distinct results.
        When a metric is not found in the DataFrame, it is represented in a no-data object.
        """
        metrics_without_data = {metric.full_name: metric for metric in metrics}

        metrics_values: list[
            Union[
                mm_schemas.ModelEndpointMonitoringMetricValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ]
        ] = []
        if not df.empty:
            grouped = df.groupby(
                [
                    mm_schemas.WriterEvent.APPLICATION_NAME,
                    mm_schemas.MetricData.METRIC_NAME,
                ],
                observed=False,
            )
        else:
            logger.debug("No metrics", missing_metrics=metrics_without_data.keys())
            grouped = []
        for (app_name, name), sub_df in grouped:
            full_name = mm_schemas.model_endpoints.compose_full_name(
                project=project,
                app=app_name,
                name=name,
                type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
            )
            metrics_values.append(
                mm_schemas.ModelEndpointMonitoringMetricValues(
                    full_name=full_name,
                    values=list(
                        zip(
                            sub_df.index,
                            sub_df[mm_schemas.MetricData.METRIC_VALUE],
                        )
                    ),  # pyright: ignore[reportArgumentType]
                )
            )
            del metrics_without_data[full_name]

        for metric in metrics_without_data.values():
            metrics_values.append(
                mm_schemas.ModelEndpointMonitoringMetricNoData(
                    full_name=metric.full_name,
                    type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
                )
            )

        return metrics_values

    @staticmethod
    def df_to_results_values(
        *,
        df: pd.DataFrame,
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric],
        project: str,
    ) -> list[
        Union[
            mm_schemas.ModelEndpointMonitoringResultValues,
            mm_schemas.ModelEndpointMonitoringMetricNoData,
        ]
    ]:
        """
        Parse a time-indexed DataFrame of results from the TSDB into a list of
        results values per distinct results.
        When a result is not found in the DataFrame, it is represented in no-data object.
        """
        metrics_without_data = {metric.full_name: metric for metric in metrics}

        metrics_values: list[
            Union[
                mm_schemas.ModelEndpointMonitoringResultValues,
                mm_schemas.ModelEndpointMonitoringMetricNoData,
            ]
        ] = []
        if not df.empty:
            grouped = df.groupby(
                [
                    mm_schemas.WriterEvent.APPLICATION_NAME,
                    mm_schemas.ResultData.RESULT_NAME,
                ],
                observed=False,
            )
        else:
            grouped = []
            logger.debug("No results", missing_results=metrics_without_data.keys())
        for (app_name, name), sub_df in grouped:
            result_kind = mlrun.model_monitoring.db.tsdb.helpers._get_result_kind(
                sub_df
            )
            full_name = mm_schemas.model_endpoints.compose_full_name(
                project=project, app=app_name, name=name
            )
            try:
                metrics_values.append(
                    mm_schemas.ModelEndpointMonitoringResultValues(
                        full_name=full_name,
                        result_kind=result_kind,
                        values=list(
                            zip(
                                sub_df.index,
                                sub_df[mm_schemas.ResultData.RESULT_VALUE],
                                sub_df[mm_schemas.ResultData.RESULT_STATUS],
                                sub_df[mm_schemas.ResultData.RESULT_EXTRA_DATA],
                            )
                        ),  # pyright: ignore[reportArgumentType]
                    )
                )
            except pydantic.v1.ValidationError:
                logger.exception(
                    "Failed to convert data-frame into `ModelEndpointMonitoringResultValues`",
                    full_name=full_name,
                    sub_df_json=sub_df.to_json(),
                )
                raise
            del metrics_without_data[full_name]

        for metric in metrics_without_data.values():
            if metric.full_name == mlrun.model_monitoring.helpers.get_invocations_fqn(
                project
            ):
                continue
            metrics_values.append(
                mm_schemas.ModelEndpointMonitoringMetricNoData(
                    full_name=metric.full_name,
                    type=mm_schemas.ModelEndpointMonitoringMetricType.RESULT,
                )
            )

        return metrics_values

    @staticmethod
    def df_to_metrics_list(
        *,
        df: pd.DataFrame,
        project: str,
        type: str,
    ) -> list[mm_schemas.ModelEndpointMonitoringMetric]:
        """
        Parse a DataFrame of metrics from the TSDB into a list of mm metrics objects.

        :param df:      The DataFrame to parse.
        :param project: The project name.
        :param type:    The type of the metrics (either "result" or "metric").

        :return:        A list of mm metrics objects.
        """

        return list(
            map(
                lambda record: mm_schemas.ModelEndpointMonitoringMetric(
                    project=project,
                    type=type,
                    app=record.get(mm_schemas.WriterEvent.APPLICATION_NAME),
                    name=record.get(mm_schemas.ResultData.RESULT_NAME)
                    or record.get(mm_schemas.MetricData.METRIC_NAME),
                    kind=record.get(mm_schemas.ResultData.RESULT_KIND),
                ),
                df.to_dict("records"),
            )
        )

    @staticmethod
    def df_to_metrics_grouped_dict(
        *,
        df: pd.DataFrame,
        project: str,
        type: str,
    ) -> dict[str, list[mm_schemas.ModelEndpointMonitoringMetric]]:
        """
        Parse a DataFrame of metrics from the TSDB into a grouped mm metrics objects by endpoint_id.

        :param df:      The DataFrame to parse.
        :param project: The project name.
        :param type:    The type of the metrics (either "result" or "metric").

        :return:        A grouped dict of mm metrics/results, using model_endpoints_ids as keys.
        """

        if df.empty:
            return {}

        grouped_by_fields = [mm_schemas.WriterEvent.APPLICATION_NAME]
        if type == "result":
            name_column = mm_schemas.ResultData.RESULT_NAME
            grouped_by_fields.append(mm_schemas.ResultData.RESULT_KIND)
        else:
            name_column = mm_schemas.MetricData.METRIC_NAME

        grouped_by_fields.append(name_column)
        # groupby has different behavior for category columns
        df["endpoint_id"] = df["endpoint_id"].astype(str)
        grouped_by_df = df.groupby("endpoint_id")
        grouped_dict = grouped_by_df.apply(
            lambda group: list(
                map(
                    lambda record: mm_schemas.ModelEndpointMonitoringMetric(
                        project=project,
                        type=type,
                        app=record.get(mm_schemas.WriterEvent.APPLICATION_NAME),
                        name=record.get(name_column),
                        **{"kind": record.get(mm_schemas.ResultData.RESULT_KIND)}
                        if type == "result"
                        else {},
                    ),
                    group[grouped_by_fields].to_dict(orient="records"),
                )
            )
        ).to_dict()
        return grouped_dict

    @staticmethod
    def df_to_events_intersection_dict(
        *,
        df: pd.DataFrame,
        project: str,
        type: Union[str, mm_schemas.ModelEndpointMonitoringMetricType],
    ) -> dict[str, list[mm_schemas.ModelEndpointMonitoringMetric]]:
        """
        Parse a DataFrame of metrics from the TSDB into a dict of intersection metrics/results by name and application
         (and kind in results).

        :param df:      The DataFrame to parse.
        :param project: The project name.
        :param type:    The type of the metrics (either "result" or "metric").

        :return:        A dictionary where the key is event type (as defined by `INTERSECT_DICT_KEYS`),
                        and the value is a list containing the intersect metrics or results across all endpoint IDs.

                        For example:
                        {
                            "intersect_metrics": [...]
                        }
        """
        dict_key = mm_schemas.INTERSECT_DICT_KEYS[type]
        metrics = []
        if df.empty:
            return {dict_key: []}

        columns_to_zip = [mm_schemas.WriterEvent.APPLICATION_NAME]

        if type == "result":
            name_column = mm_schemas.ResultData.RESULT_NAME
            columns_to_zip.append(mm_schemas.ResultData.RESULT_KIND)
        else:
            name_column = mm_schemas.MetricData.METRIC_NAME
        columns_to_zip.insert(1, name_column)

        # groupby has different behavior for category columns
        df["endpoint_id"] = df["endpoint_id"].astype(str)
        df["event_values"] = list(zip(*[df[col] for col in columns_to_zip]))
        grouped_by_event_values = df.groupby("endpoint_id")["event_values"].apply(set)
        common_event_values_combinations = set.intersection(*grouped_by_event_values)
        result_kind = None
        for data in common_event_values_combinations:
            application_name, event_name = data[0], data[1]
            if len(data) > 2:  # in result case
                result_kind = data[2]
            metrics.append(
                mm_schemas.ModelEndpointMonitoringMetric(
                    project=project,
                    type=type,
                    app=application_name,
                    name=event_name,
                    kind=result_kind,
                )
            )
        return {dict_key: metrics}

    @staticmethod
    def _get_start_end(
        start: Union[datetime, None],
        end: Union[datetime, None],
    ) -> tuple[datetime, datetime]:
        """
        static utils function for tsdb start end format
        :param start:       Either None or datetime, None is handled as datetime.min(tz=timezone.utc)
        :param end:         Either None or datetime, None is handled as datetime.now(tz=timezone.utc)
        :return:            start datetime, end datetime
        """
        start = start or mlrun.utils.datetime_min()
        end = end or mlrun.utils.datetime_now()
        if not (isinstance(start, datetime) and isinstance(end, datetime)):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Both start and end must be datetime objects"
            )
        return start, end
