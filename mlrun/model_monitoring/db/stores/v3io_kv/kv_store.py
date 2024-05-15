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

import json
import os
import typing
from dataclasses import dataclass
from http import HTTPStatus

import v3io.dataplane
import v3io.dataplane.response

import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints
import mlrun.model_monitoring.db
import mlrun.utils.v3io_clients
from mlrun.utils import logger

# Fields to encode before storing in the KV table or to decode after retrieving
fields_to_encode_decode = [
    mm_constants.EventFieldType.FEATURE_STATS,
    mm_constants.EventFieldType.CURRENT_STATS,
]

_METRIC_FIELDS: list[str] = [
    mm_constants.WriterEvent.APPLICATION_NAME,
    mm_constants.MetricData.METRIC_NAME,
    mm_constants.MetricData.METRIC_VALUE,
    mm_constants.WriterEvent.START_INFER_TIME,
    mm_constants.WriterEvent.END_INFER_TIME,
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
        name=mm_constants.ResultData.RESULT_NAME,
        type=mm_constants.GrafanaColumnType.STRING,
        nullable=False,
    )
]

_METRIC_SCHEMA: list[SchemaField] = [
    SchemaField(
        name=mm_constants.WriterEvent.APPLICATION_NAME,
        type=mm_constants.GrafanaColumnType.STRING,
        nullable=False,
    ),
    SchemaField(
        name=mm_constants.MetricData.METRIC_NAME,
        type=mm_constants.GrafanaColumnType.STRING,
        nullable=False,
    ),
]


_KIND_TO_SCHEMA_PARAMS: dict[mm_constants.WriterEventKind, SchemaParams] = {
    mm_constants.WriterEventKind.RESULT: SchemaParams(
        key=mm_constants.WriterEvent.APPLICATION_NAME, fields=_RESULT_SCHEMA
    ),
    mm_constants.WriterEventKind.METRIC: SchemaParams(
        key="metric_id", fields=_METRIC_SCHEMA
    ),
}


class KVStoreBase(mlrun.model_monitoring.db.StoreBase):
    """
    Handles the DB operations when the DB target is from type KV. For the KV operations, we use an instance of V3IO
    client and usually the KV table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/.
    """

    def __init__(self, project: str, access_key: typing.Optional[str] = None) -> None:
        super().__init__(project=project)
        # Initialize a V3IO client instance
        self.access_key = access_key or os.environ.get("V3IO_ACCESS_KEY")
        self.client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api, access_key=self.access_key
        )
        # Get the KV table path and container
        self.path, self.container = self._get_path_and_container()

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
            key=endpoint[mm_constants.EventFieldType.UID],
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
            access_key=self.access_key,
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
            kind=mm_constants.ModelMonitoringStoreKinds.ENDPOINTS,
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
    ) -> list[dict[str, typing.Any]]:
        """
        Returns a list of model endpoint dictionaries, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available model endpoints for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key=value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param uids:            List of model endpoint unique ids to include in the result.


        :return: A list of model endpoint dictionaries.
        """

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
                    labels,
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
                if mm_constants.EventFieldType.UID not in item:
                    # This is kept for backwards compatibility - in old versions the key column named endpoint_id
                    uids.append(item[mm_constants.EventFieldType.ENDPOINT_ID])
                else:
                    uids.append(item[mm_constants.EventFieldType.UID])

        # Add each relevant model endpoint to the model endpoints list
        for endpoint_id in uids:
            endpoint = self.get_model_endpoint(
                endpoint_id=endpoint_id,
            )
            endpoint_list.append(endpoint)

        return endpoint_list

    def delete_model_endpoints_resources(self):
        """
        Delete all model endpoints resources in V3IO KV.
        """

        endpoints = self.list_model_endpoints()

        # Delete model endpoint record from KV table
        for endpoint_dict in endpoints:
            if mm_constants.EventFieldType.UID not in endpoint_dict:
                # This is kept for backwards compatibility - in old versions the key column named endpoint_id
                endpoint_id = endpoint_dict[mm_constants.EventFieldType.ENDPOINT_ID]
            else:
                endpoint_id = endpoint_dict[mm_constants.EventFieldType.UID]
            self.delete_model_endpoint(
                endpoint_id,
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

    def write_application_event(
        self,
        event: dict[str, typing.Any],
        kind: mm_constants.WriterEventKind = mm_constants.WriterEventKind.RESULT,
    ) -> None:
        """
        Write a new application event in the target table.

        :param event: An event dictionary that represents the application result, should be corresponded to the
                      schema defined in the :py:class:`~mlrun.common.schemas.model_monitoring.constants.WriterEvent`
                      object.
        :param kind: The type of the event, can be either "result" or "metric".
        """

        container = self.get_v3io_monitoring_apps_container(project_name=self.project)
        endpoint_id = event.pop(mm_constants.WriterEvent.ENDPOINT_ID)

        if kind == mm_constants.WriterEventKind.METRIC:
            table_path = f"{endpoint_id}_metrics"
            key = f"{event[mm_constants.WriterEvent.APPLICATION_NAME]}.{event[mm_constants.MetricData.METRIC_NAME]}"
            attributes = {event_key: event[event_key] for event_key in _METRIC_FIELDS}
        elif kind == mm_constants.WriterEventKind.RESULT:
            table_path = endpoint_id
            key = event.pop(mm_constants.WriterEvent.APPLICATION_NAME)
            metric_name = event.pop(mm_constants.ResultData.RESULT_NAME)
            attributes = {metric_name: json.dumps(event)}
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
        self, *, container: str, table_path: str, kind: mm_constants.WriterEventKind
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
            data = self.client.kv.get(
                container=self._get_monitoring_schedules_container(
                    project_name=self.project
                ),
                table_path=endpoint_id,
                key=application_name,
            )
            return data.output.item[mm_constants.SchedulingKeys.LAST_ANALYZED]
        except v3io.dataplane.response.HttpResponseError as err:
            logger.debug("Error while getting last analyzed time", err=err)
            raise mlrun.errors.MLRunNotFoundError(
                f"No last analyzed value has been found for {application_name} "
                f"that processes model endpoint {endpoint_id}",
            )

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
            attributes={mm_constants.SchedulingKeys.LAST_ANALYZED: last_analyzed},
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
                kind=mm_constants.ModelMonitoringStoreKinds.EVENTS,
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
            token=self.access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=self.container,
        )

    @staticmethod
    def _build_kv_cursor_filter_expression(
        project: str,
        function: str = None,
        model: str = None,
        labels: list[str] = None,
        top_level: bool = False,
    ) -> str:
        """
        Convert the provided filters into a valid filter expression. The expected filter expression includes different
        conditions, divided by ' AND '.

        :param project:    The name of the project.
        :param model:      The name of the model to filter by.
        :param function:   The name of the function to filter by.
        :param labels:     A list of labels to filter by. Label filters work by either filtering a specific value of
                           a label (i.e. list("key=value")) or by looking for the existence of a given
                           key (i.e. "key").
        :param top_level:  If True will return only routers and endpoint that are NOT children of any router.

        :return: A valid filter expression as a string.

        :raise MLRunInvalidArgumentError: If project value is None.
        """

        if not project:
            raise mlrun.errors.MLRunInvalidArgumentError("project can't be empty")

        # Add project filter
        filter_expression = [f"project=='{project}'"]

        # Add function and model filters
        if function:
            filter_expression.append(f"function=='{function}'")
        if model:
            filter_expression.append(f"model=='{model}'")

        # Add labels filters
        if labels:
            for label in labels:
                if not label.startswith("_"):
                    label = f"_{label}"

                if "=" in label:
                    lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                    filter_expression.append(f"{lbl}=='{value}'")
                else:
                    filter_expression.append(f"exists({label})")

        # Apply top_level filter (remove endpoints that considered a child of a router)
        if top_level:
            filter_expression.append(
                f"(endpoint_type=='{str(mm_constants.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mm_constants.EndpointType.ROUTER.value)}')"
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
            mm_constants.EventFieldType.ERROR_COUNT in endpoint
            and endpoint[mm_constants.EventFieldType.ERROR_COUNT] == "null"
        ):
            endpoint[mm_constants.EventFieldType.ERROR_COUNT] = "0"

        # Validate default value for `metrics`
        # For backwards compatibility reasons, we validate that the model endpoint includes the `metrics` key
        if (
            mm_constants.EventFieldType.METRICS in endpoint
            and endpoint[mm_constants.EventFieldType.METRICS] == "null"
        ):
            endpoint[mm_constants.EventFieldType.METRICS] = json.dumps(
                {
                    mm_constants.EventKeyMetrics.GENERIC: {
                        mm_constants.EventLiveStats.LATENCY_AVG_1H: 0,
                        mm_constants.EventLiveStats.PREDICTIONS_PER_SECOND: 0,
                    }
                }
            )
        # Validate key `uid` instead of `endpoint_id`
        # For backwards compatibility reasons, we replace the `endpoint_id` with `uid` which is the updated key name
        if mm_constants.EventFieldType.ENDPOINT_ID in endpoint:
            endpoint[mm_constants.EventFieldType.UID] = endpoint[
                mm_constants.EventFieldType.ENDPOINT_ID
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

    def _extract_metrics_from_items(
        self, app_items: list[dict[str, str]]
    ) -> list[mm_constants.ModelEndpointMonitoringMetric]:
        metrics: list[mm_constants.ModelEndpointMonitoringMetric] = []
        for app_item in app_items:
            # See https://www.iguazio.com/docs/latest-release/services/data-layer/reference/system-attributes/#sys-attr-__name
            app_name = app_item.pop("__name")
            if app_name == ".#schema":
                continue
            for result_name in app_item:
                metrics.append(
                    mm_constants.ModelEndpointMonitoringMetric(
                        project=self.project,
                        app=app_name,
                        type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
                        name=result_name,
                        full_name=mlrun.common.schemas.model_monitoring.model_endpoints._compose_full_name(
                            project=self.project, app=app_name, name=result_name
                        ),
                    )
                )
        return metrics

    def get_model_endpoint_metrics(
        self, endpoint_id: str
    ) -> list[mm_constants.ModelEndpointMonitoringMetric]:
        """Get model monitoring results and metrics on the endpoint"""
        metrics: list[mm_constants.ModelEndpointMonitoringMetric] = []
        container = self.get_v3io_monitoring_apps_container(self.project)
        try:
            response = self.client.kv.scan(container=container, table_path=endpoint_id)
        except v3io.dataplane.response.HttpResponseError as err:
            if err.status_code == HTTPStatus.NOT_FOUND:
                logger.warning(
                    "Attempt getting metrics and results - no data. Check the "
                    "project name, endpoint, or wait for the applications to start.",
                    container=container,
                    table_path=endpoint_id,
                )
                return []
            raise

        while True:
            metrics.extend(self._extract_metrics_from_items(response.output.items))
            if response.output.last:
                break
            # TODO: Use AIO client: `v3io.aio.dataplane.client.Client`
            response = self.client.kv.scan(
                container=container,
                table_path=endpoint_id,
                marker=response.output.next_marker,
            )

        return metrics
