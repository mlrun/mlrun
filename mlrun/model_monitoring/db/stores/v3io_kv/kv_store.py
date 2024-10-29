# Copyright 2023 Iguazio
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
import http
import json
import typing
from dataclasses import dataclass
from http import HTTPStatus

import v3io.dataplane
import v3io.dataplane.output
import v3io.dataplane.response
from v3io.dataplane import Client as V3IOClient

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring as mm_schemas
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.db import StoreBase
from mlrun.utils import logger

# Fields to encode before storing in the KV table or to decode after retrieving
fields_to_encode_decode = [
    mm_schemas.EventFieldType.FEATURE_STATS,
    mm_schemas.EventFieldType.CURRENT_STATS,
]

_METRIC_FIELDS: list[str] = [
    mm_schemas.WriterEvent.APPLICATION_NAME.value,
    mm_schemas.MetricData.METRIC_NAME.value,
    mm_schemas.MetricData.METRIC_VALUE.value,
    mm_schemas.WriterEvent.START_INFER_TIME.value,
    mm_schemas.WriterEvent.END_INFER_TIME.value,
]


class SchemaField(typing.TypedDict):
    name: str
    type: str
    nullable: bool


@dataclass
class SchemaParams:
    key: str
    fields: list[SchemaField]


_RESULT_SCHEMA: list[SchemaField] = [
    SchemaField(
        name=mm_schemas.ResultData.RESULT_NAME,
        type=mm_schemas.GrafanaColumnType.STRING,
        nullable=False,
    )
]

_METRIC_SCHEMA: list[SchemaField] = [
    SchemaField(
        name=mm_schemas.WriterEvent.APPLICATION_NAME,
        type=mm_schemas.GrafanaColumnType.STRING,
        nullable=False,
    ),
    SchemaField(
        name=mm_schemas.MetricData.METRIC_NAME,
        type=mm_schemas.GrafanaColumnType.STRING,
        nullable=False,
    ),
]


_KIND_TO_SCHEMA_PARAMS: dict[mm_schemas.WriterEventKind, SchemaParams] = {
    mm_schemas.WriterEventKind.RESULT: SchemaParams(
        key=mm_schemas.WriterEvent.APPLICATION_NAME, fields=_RESULT_SCHEMA
    ),
    mm_schemas.WriterEventKind.METRIC: SchemaParams(
        key="metric_id", fields=_METRIC_SCHEMA
    ),
}

_EXCLUDE_SCHEMA_FILTER_EXPRESSION = '__name!=".#schema"'


class KVStoreBase(StoreBase):
    type: typing.ClassVar[str] = "v3io-nosql"
    """
    Handles the DB operations when the DB target is from type KV. For the KV operations, we use an instance of V3IO
    client and usually the KV table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/.
    """

    def __init__(
        self,
        project: str,
    ) -> None:
        super().__init__(project=project)
        self._client = None
        # Get the KV table path and container
        self.path, self.container = self._get_path_and_container()

    @property
    def client(self) -> V3IOClient:
        if not self._client:
            self._client = mlrun.utils.v3io_clients.get_v3io_client(
                endpoint=mlrun.mlconf.v3io_api,
            )
        return self._client

    def write_model_endpoint(self, endpoint: dict[str, typing.Any]):
        """
        Create a new endpoint record in the KV table.

        :param endpoint: model endpoint dictionary that will be written into the DB.
        """

        for field in fields_to_encode_decode:
            if field in endpoint:
                # Encode to binary data
                endpoint[field] = self._encode_field(endpoint[field])

        self.client.kv.put(
            container=self.container,
            table_path=self.path,
            key=endpoint[mm_schemas.EventFieldType.UID],
            attributes=endpoint,
        )

        self._infer_kv_schema()

    def update_model_endpoint(
        self, endpoint_id: str, attributes: dict[str, typing.Any]
    ):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the KV table.

        """

        for field in fields_to_encode_decode:
            if field in attributes:
                # Encode to binary data
                attributes[field] = self._encode_field(attributes[field])

        self.client.kv.update(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
            attributes=attributes,
        )

    def delete_model_endpoint(
        self,
        endpoint_id: str,
    ):
        """
        Deletes the KV record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """

        self.client.kv.delete(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
        )

    def get_model_endpoint(
        self,
        endpoint_id: str,
    ) -> dict[str, typing.Any]:
        """
        Get a single model endpoint record.

        :param endpoint_id: The unique id of the model endpoint.

        :return: A model endpoint record as a dictionary.

        :raise MLRunNotFoundError: If the endpoint was not found.
        """

        # Getting the raw data from the KV table
        endpoint = self.client.kv.get(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        )
        endpoint = endpoint.output.item

        for field in fields_to_encode_decode:
            if field in endpoint:
                # Decode binary data
                endpoint[field] = self._decode_field(endpoint[field])

        if not endpoint:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # For backwards compatability: replace null values for `error_count` and `metrics`
        self.validate_old_schema_fields(endpoint=endpoint)

        return endpoint

    def _get_path_and_container(self):
        """Getting path and container based on the model monitoring configurations"""
        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project,
            kind=mm_schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            path
        )
        return path, container

    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: list[str] = None,
        top_level: bool = None,
        uids: list = None,
        include_stats: bool = None,
    ) -> list[dict[str, typing.Any]]:
        # # Initialize an empty model endpoints list
        endpoint_list = []

        # Retrieve the raw data from the KV table and get the endpoint ids
        try:
            cursor = self.client.kv.new_cursor(
                container=self.container,
                table_path=self.path,
                filter_expression=self._build_kv_cursor_filter_expression(
                    self.project,
                    function,
                    model,
                    top_level,
                ),
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )
            items = cursor.all()

        except Exception as exc:
            logger.warning(
                "Failed retrieving raw data from kv table",
                exc=mlrun.errors.err_to_str(exc),
            )
            return endpoint_list
        # Create a list of model endpoints unique ids
        if uids is None:
            uids = []
            for item in items:
                if mm_schemas.EventFieldType.UID not in item:
                    # This is kept for backwards compatibility - in old versions the key column named endpoint_id
                    uids.append(item[mm_schemas.EventFieldType.ENDPOINT_ID])
                else:
                    uids.append(item[mm_schemas.EventFieldType.UID])

        # Add each relevant model endpoint to the model endpoints list
        for endpoint_id in uids:
            endpoint_dict = self.get_model_endpoint(
                endpoint_id=endpoint_id,
            )
            if not include_stats:
                # Exclude these fields when listing model endpoints to avoid returning too much data (ML-6594)
                endpoint_dict.pop(mm_schemas.EventFieldType.FEATURE_STATS)
                endpoint_dict.pop(mm_schemas.EventFieldType.CURRENT_STATS)

            if labels and not self._validate_labels(
                endpoint_dict=endpoint_dict, labels=labels
            ):
                continue

            endpoint_list.append(endpoint_dict)

        return endpoint_list

    def delete_model_endpoints_resources(self):
        """
        Delete all model endpoints resources in V3IO KV.
        """
        logger.debug(
            "Deleting model monitoring endpoints resources in V3IO KV",
            project=self.project,
        )

        endpoints = self.list_model_endpoints()

        # Delete model endpoint record from KV table
        for endpoint_dict in endpoints:
            if mm_schemas.EventFieldType.UID not in endpoint_dict:
                # This is kept for backwards compatibility - in old versions the key column named endpoint_id
                endpoint_id = endpoint_dict[mm_schemas.EventFieldType.ENDPOINT_ID]
            else:
                endpoint_id = endpoint_dict[mm_schemas.EventFieldType.UID]

            logger.debug(
                "Deleting model endpoint resources from the V3IO KV table",
                endpoint_id=endpoint_id,
                project=self.project,
            )

            self.delete_model_endpoint(
                endpoint_id,
            )

        logger.debug(
            "Successfully deleted model monitoring endpoints from the V3IO KV table",
            project=self.project,
        )

        # Delete remain records in the KV
        all_records = self.client.kv.new_cursor(
            container=self.container,
            table_path=self.path,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        ).all()

        all_records = [r["__name"] for r in all_records]

        # Cleanup KV
        for record in all_records:
            self.client.kv.delete(
                container=self.container,
                table_path=self.path,
                key=record,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )

    @staticmethod
    def _get_results_table_path(endpoint_id: str) -> str:
        return endpoint_id

    @staticmethod
    def _get_metrics_table_path(endpoint_id: str) -> str:
        return f"{endpoint_id}_metrics"

    def write_application_event(
        self,
        event: dict[str, typing.Any],
        kind: mm_schemas.WriterEventKind = mm_schemas.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a new application event in the target table.

        :param event: An event dictionary that represents the application result, should be corresponded to the
                      schema defined in the :py:class:`~mlrun.common.schemas.model_monitoring.constants.WriterEvent`
                      object.
        :param kind: The type of the event, can be either "result" or "metric".
        """

        container = self.get_v3io_monitoring_apps_container(project_name=self.project)
        endpoint_id = event.pop(mm_schemas.WriterEvent.ENDPOINT_ID)

        if kind == mm_schemas.WriterEventKind.METRIC:
            table_path = self._get_metrics_table_path(endpoint_id)
            key = f"{event[mm_schemas.WriterEvent.APPLICATION_NAME]}.{event[mm_schemas.MetricData.METRIC_NAME]}"
            attributes = {event_key: event[event_key] for event_key in _METRIC_FIELDS}
        elif kind == mm_schemas.WriterEventKind.RESULT:
            table_path = self._get_results_table_path(endpoint_id)
            key = event.pop(mm_schemas.WriterEvent.APPLICATION_NAME)
            metric_name = event.pop(mm_schemas.ResultData.RESULT_NAME)
            attributes = {metric_name: self._encode_field(json.dumps(event))}
        else:
            raise ValueError(f"Invalid {kind = }")

        self.client.kv.update(
            container=container,
            table_path=table_path,
            key=key,
            attributes=attributes,
        )

        schema_file = self.client.kv.new_cursor(
            container=container,
            table_path=table_path,
            filter_expression='__name==".#schema"',
        )

        if not schema_file.all():
            logger.info(
                "Generating a new V3IO KV schema file",
                container=container,
                table_path=table_path,
            )
            self._generate_kv_schema(
                container=container, table_path=table_path, kind=kind
            )
        logger.info("Updated V3IO KV successfully", key=key)

    def _generate_kv_schema(
        self, *, container: str, table_path: str, kind: mm_schemas.WriterEventKind
    ) -> None:
        """Generate V3IO KV schema file which will be used by the model monitoring applications dashboard in Grafana."""
        schema_params = _KIND_TO_SCHEMA_PARAMS[kind]
        res = self.client.kv.create_schema(
            container=container,
            table_path=table_path,
            key=schema_params.key,
            fields=schema_params.fields,
        )
        if res.status_code != HTTPStatus.OK:
            raise mlrun.errors.MLRunBadRequestError(
                f"Couldn't infer schema for endpoint {table_path} which is required for Grafana dashboards"
            )
        else:
            logger.info("Generated V3IO KV schema successfully", table_path=table_path)

    def get_last_analyzed(self, endpoint_id: str, application_name: str) -> int:
        """
        Get the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.

        :return: Timestamp as a Unix time.
        :raise:  MLRunNotFoundError if last analyzed value is not found.

        """
        try:
            response = self.client.kv.get(
                container=self._get_monitoring_schedules_container(
                    project_name=self.project
                ),
                table_path=endpoint_id,
                key=application_name,
            )
            return response.output.item[mm_schemas.SchedulingKeys.LAST_ANALYZED]
        except v3io.dataplane.response.HttpResponseError as err:
            if err.status_code == http.HTTPStatus.NOT_FOUND:
                logger.debug("Last analyzed time not found", err=err)
                raise mlrun.errors.MLRunNotFoundError(
                    f"No last analyzed value has been found for {application_name} "
                    f"that processes model endpoint {endpoint_id}",
                )
            logger.error("Error while getting last analyzed time", err=err)
            raise err

    def update_last_analyzed(
        self, endpoint_id: str, application_name: str, last_analyzed: int
    ):
        """
        Update the last analyzed time for the provided model endpoint and application.

        :param endpoint_id:      The unique id of the model endpoint.
        :param application_name: Registered application name.
        :param last_analyzed:    Timestamp as a Unix time that represents the last analyzed time of a certain
                                 application and model endpoint.
        """
        self.client.kv.put(
            container=self._get_monitoring_schedules_container(
                project_name=self.project
            ),
            table_path=endpoint_id,
            key=application_name,
            attributes={mm_schemas.SchedulingKeys.LAST_ANALYZED: last_analyzed},
        )

    def _generate_tsdb_paths(self) -> tuple[str, str]:
        """Generate a short path to the TSDB resources and a filtered path for the frames object
        :return: A tuple of:
             [0] = Short path to the TSDB resources
             [1] = Filtered path to TSDB events without schema and container
        """
        # Full path for the time series DB events
        full_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mm_schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )

        # Generate the main directory with the TSDB resources
        tsdb_path = (
            mlrun.common.model_monitoring.helpers.parse_model_endpoint_project_prefix(
                full_path, self.project
            )
        )

        # Generate filtered path without schema and container as required by the frames object
        (
            _,
            _,
            filtered_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            full_path
        )
        return tsdb_path, filtered_path

    def _infer_kv_schema(self):
        """
        Create KV schema file if not exist. This schema is being used by the Grafana dashboards.
        """

        schema_file = self.client.kv.new_cursor(
            container=self.container,
            table_path=self.path,
            filter_expression='__name==".#schema"',
        )

        if not schema_file.all():
            logger.info("Generate a new V3IO KV schema file", kv_table_path=self.path)
            frames_client = self._get_frames_client()
            frames_client.execute(backend="kv", table=self.path, command="infer_schema")

    def _get_frames_client(self):
        return mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=self.container,
        )

    @staticmethod
    def _build_kv_cursor_filter_expression(
        project: str,
        function: str = None,
        model: str = None,
        top_level: bool = False,
    ) -> str:
        """
        Convert the provided filters into a valid filter expression. The expected filter expression includes different
        conditions, divided by ' AND '.

        :param project:         The name of the project.
        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.

        :return: A valid filter expression as a string.

        :raise MLRunInvalidArgumentError: If project value is None.
        """

        if not project:
            raise mlrun.errors.MLRunInvalidArgumentError("project can't be empty")

        # Add project filter
        filter_expression = [f"{mm_schemas.EventFieldType.PROJECT}=='{project}'"]

        # Add function and model filters
        if function:
            function_uri = f"{project}/{function}" if function else None
            filter_expression.append(
                f"{mm_schemas.EventFieldType.FUNCTION_URI}=='{function_uri}'"
            )
        if model:
            model = model if ":" in model else f"{model}:latest"
            filter_expression.append(f"{mm_schemas.EventFieldType.MODEL}=='{model}'")

        # Apply top_level filter (remove endpoints that considered a child of a router)
        if top_level:
            filter_expression.append(
                f"(endpoint_type=='{str(mm_schemas.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mm_schemas.EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)

    @staticmethod
    def validate_old_schema_fields(endpoint: dict):
        """
        Replace default null values for `error_count` and `metrics` for users that logged a model endpoint before 1.3.0.
        In addition, this function also validates that the key name of the endpoint unique id is `uid` and not
        `endpoint_id` that has been used before 1.3.0.

        Leaving here for backwards compatibility which related to the model endpoint schema.

        :param endpoint: An endpoint flattened dictionary.
        """

        # Validate default value for `error_count`
        # For backwards compatibility reasons, we validate that the model endpoint includes the `error_count` key
        if (
            mm_schemas.EventFieldType.ERROR_COUNT in endpoint
            and endpoint[mm_schemas.EventFieldType.ERROR_COUNT] == "null"
        ):
            endpoint[mm_schemas.EventFieldType.ERROR_COUNT] = "0"

        # Validate default value for `metrics`
        # For backwards compatibility reasons, we validate that the model endpoint includes the `metrics` key
        if (
            mm_schemas.EventFieldType.METRICS in endpoint
            and endpoint[mm_schemas.EventFieldType.METRICS] == "null"
        ):
            endpoint[mm_schemas.EventFieldType.METRICS] = json.dumps(
                {
                    mm_schemas.EventKeyMetrics.GENERIC: {
                        mm_schemas.EventLiveStats.LATENCY_AVG_1H: 0,
                        mm_schemas.EventLiveStats.PREDICTIONS_PER_SECOND: 0,
                    }
                }
            )
        # Validate key `uid` instead of `endpoint_id`
        # For backwards compatibility reasons, we replace the `endpoint_id` with `uid` which is the updated key name
        if mm_schemas.EventFieldType.ENDPOINT_ID in endpoint:
            endpoint[mm_schemas.EventFieldType.UID] = endpoint[
                mm_schemas.EventFieldType.ENDPOINT_ID
            ]

    @staticmethod
    def _encode_field(field: typing.Union[str, bytes]) -> bytes:
        """Encode a provided field. Mainly used when storing data in the KV table."""

        if isinstance(field, str):
            return field.encode("ascii")
        return field

    @staticmethod
    def _decode_field(field: typing.Union[str, bytes]) -> str:
        """Decode a provided field. Mainly used when retrieving data from the KV table."""

        if isinstance(field, bytes):
            return field.decode()
        return field

    @staticmethod
    def get_v3io_monitoring_apps_container(project_name: str) -> str:
        return f"users/pipelines/{project_name}/monitoring-apps"

    @staticmethod
    def _get_monitoring_schedules_container(project_name: str) -> str:
        return f"users/pipelines/{project_name}/monitoring-schedules/functions"

    def _extract_results_from_items(
        self, app_items: list[dict[str, str]]
    ) -> list[mm_schemas.ModelEndpointMonitoringMetric]:
        """Assuming .#schema items are filtered out"""
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric] = []
        for app_item in app_items:
            app_name = app_item.pop("__name")
            for result_name in app_item:
                metrics.append(
                    mm_schemas.ModelEndpointMonitoringMetric(
                        project=self.project,
                        app=app_name,
                        type=mm_schemas.ModelEndpointMonitoringMetricType.RESULT,
                        name=result_name,
                        full_name=mm_schemas.model_endpoints._compose_full_name(
                            project=self.project, app=app_name, name=result_name
                        ),
                    )
                )
        return metrics

    def _extract_metrics_from_items(
        self, result_items: list[dict[str, str]]
    ) -> list[mm_schemas.ModelEndpointMonitoringMetric]:
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric] = []
        logger.debug("Result items", result_items=result_items)
        for result_item in result_items:
            app = result_item[mm_schemas.WriterEvent.APPLICATION_NAME]
            name = result_item[mm_schemas.MetricData.METRIC_NAME]
            metrics.append(
                mm_schemas.ModelEndpointMonitoringMetric(
                    project=self.project,
                    app=app,
                    type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
                    name=name,
                    full_name=mm_schemas.model_endpoints._compose_full_name(
                        project=self.project,
                        app=app,
                        name=name,
                        type=mm_schemas.ModelEndpointMonitoringMetricType.METRIC,
                    ),
                )
            )
        return metrics

    def get_model_endpoint_metrics(
        self, endpoint_id: str, type: mm_schemas.ModelEndpointMonitoringMetricType
    ) -> list[mm_schemas.ModelEndpointMonitoringMetric]:
        """Get model monitoring results and metrics on the endpoint"""
        metrics: list[mm_schemas.ModelEndpointMonitoringMetric] = []
        container = self.get_v3io_monitoring_apps_container(self.project)
        if type == mm_schemas.ModelEndpointMonitoringMetricType.METRIC:
            table_path = self._get_metrics_table_path(endpoint_id)
            items_extractor = self._extract_metrics_from_items
        elif type == mm_schemas.ModelEndpointMonitoringMetricType.RESULT:
            table_path = self._get_results_table_path(endpoint_id)
            items_extractor = self._extract_results_from_items
        else:
            raise ValueError(f"Invalid metric {type = }")

        def scan(
            marker: typing.Optional[str] = None,
        ) -> v3io.dataplane.response.Response:
            # TODO: Use AIO client: `v3io.aio.dataplane.client.Client`
            return self.client.kv.scan(
                container=container,
                table_path=table_path,
                marker=marker,
                filter_expression=_EXCLUDE_SCHEMA_FILTER_EXPRESSION,
            )

        try:
            response = scan()
        except v3io.dataplane.response.HttpResponseError as err:
            if err.status_code == HTTPStatus.NOT_FOUND:
                logger.warning(
                    f"Attempt getting {type}s - no data. Check the "
                    "project name, endpoint, or wait for the applications to start.",
                    container=container,
                    table_path=table_path,
                )
                return []
            raise

        while True:
            output = typing.cast(v3io.dataplane.output.GetItemsOutput, response.output)
            metrics.extend(items_extractor(output.items))
            if output.last:
                break
            response = scan(marker=output.next_marker)

        return metrics
