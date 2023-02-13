# Copyright 2018 Iguazio
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
#

import enum
import json
import typing
from abc import ABC, abstractmethod

import v3io.dataplane
import v3io_frames

import mlrun
import mlrun.api.schemas
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.utils.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.utils import logger


class _ModelEndpointStore(ABC):
    """
    An abstract class to handle the model endpoint in the DB target.
    """

    def __init__(self, project: str):
        """
        Initialize a new model endpoint target.

        :param project:             The name of the project.
        """
        self.project = project

    @abstractmethod
    def write_model_endpoint(self, endpoint: mlrun.api.schemas.ModelEndpoint):
        """
        Create a new endpoint record in the DB table.

        :param endpoint: ModelEndpoint object that will be written into the DB.
        """
        pass

    @abstractmethod
    def update_model_endpoint(self, endpoint_id: str, attributes: dict):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the KV table.

        """
        pass

    @abstractmethod
    def delete_model_endpoint(self, endpoint_id: str):
        """
        Deletes the record of a given model endpoint id.

        :param endpoint_id: The unique id of the model endpoint.
        """
        pass

    @abstractmethod
    def delete_model_endpoints_resources(
        self, endpoints: mlrun.api.schemas.model_endpoints.ModelEndpointList
    ):
        """
        Delete all model endpoints resources.

        :param endpoints: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
        """
        pass

    @abstractmethod
    def get_model_endpoint(
        self,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
        endpoint_id: str = None,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Get a single model endpoint object. You can apply different time series metrics that will be added to the
           result.

        :param endpoint_id:      The unique id of the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param metrics:          A list of metrics to return for the model endpoint. There are pre-defined metrics for
                                 model endpoints such as predictions_per_second and latency_avg_5m but also custom
                                 metrics defined by the user. Please note that these metrics are stored in the time
                                 series DB and the results will be appeared under model_endpoint.spec.metrics.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A ModelEndpoint object.
        """
        pass

    @abstractmethod
    def list_model_endpoints(
        self, model: str, function: str, labels: typing.List, top_level: bool
    ):
        """
        Returns a list of endpoint unique ids, supports filtering by model, function,
        labels or top level. By default, when no filters are applied, all available endpoint ids for the given project
        will be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key==value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.

        :return: List of model endpoints unique ids.
        """
        pass


class _ModelEndpointKVStore(_ModelEndpointStore):
    """
    Handles the DB operations when the DB target is from type KV. For the KV operations, we use an instance of V3IO
    client and usually the KV table can be found under v3io:///users/pipelines/project-name/model-endpoints/endpoints/.
    """

    def __init__(self, access_key: str, project: str):
        super().__init__(project=project)
        # Initialize a V3IO client instance
        self.access_key = access_key
        self.client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api, access_key=self.access_key
        )
        # Get the KV table path and container
        self.path, self.container = self._get_path_and_container()

    def write_model_endpoint(self, endpoint: mlrun.api.schemas.ModelEndpoint):
        """
        Create a new endpoint record in the KV table.

        :param endpoint: ModelEndpoint object that will be written into the DB.
        """

        # Flatten the model endpoint structure in order to write it into the DB table.
        # More details about the model endpoint available attributes can be found under
        # :py:class:`~mlrun.api.schemas.ModelEndpoint`.`
        attributes = self.flatten_model_endpoint_attributes(endpoint)

        # Create or update the model endpoint record
        self.client.kv.put(
            container=self.container,
            table_path=self.path,
            key=endpoint.metadata.uid,
            attributes=attributes,
        )

    def update_model_endpoint(self, endpoint_id: str, attributes: dict):
        """
        Update a model endpoint record with a given attributes.

        :param endpoint_id: The unique id of the model endpoint.
        :param attributes: Dictionary of attributes that will be used for update the model endpoint. Note that the keys
                           of the attributes dictionary should exist in the KV table. More details about the model
                           endpoint available attributes can be found under
                           :py:class:`~mlrun.api.schemas.ModelEndpoint`.

        """

        self.client.kv.update(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
            attributes=attributes,
        )

        logger.info("Model endpoint table updated", endpoint_id=endpoint_id)

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

        logger.info("Model endpoint table cleared", endpoint_id=endpoint_id)

    def get_model_endpoint(
        self,
        endpoint_id: str = None,
        start: str = "now-1h",
        end: str = "now",
        metrics: typing.List[str] = None,
        feature_analysis: bool = False,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Get a single model endpoint object. You can apply different time series metrics that will be added to the
        result.

        :param endpoint_id:      The unique id of the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param metrics:          A list of metrics to return for the model endpoint. There are pre-defined metrics for
                                 model endpoints such as predictions_per_second and latency_avg_5m but also custom
                                 metrics defined by the user. Please note that these metrics are stored in the time
                                 series DB and the results will be appeared under model_endpoint.spec.metrics.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A ModelEndpoint object.
        """
        logger.info(
            "Getting model endpoint record from kv",
            endpoint_id=endpoint_id,
        )

        # Getting the raw data from the KV table
        endpoint = self.client.kv.get(
            container=self.container,
            table_path=self.path,
            key=endpoint_id,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
            access_key=self.access_key,
        )
        endpoint = endpoint.output.item

        if not endpoint:
            raise mlrun.errors.MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

        # Generate a model endpoint object from the model endpoint KV record
        endpoint_obj = self._convert_into_model_endpoint_object(
            endpoint, start, end, metrics, feature_analysis
        )

        return endpoint_obj

    def _convert_into_model_endpoint_object(
        self, endpoint, start, end, metrics, feature_analysis
    ):
        """
        Create a ModelEndpoint object according to a provided endpoint record from the DB.

        :param endpoint:         KV record of model endpoint which need to be converted into a valid ModelEndpoint
                                 object.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param metrics:          A list of metrics to return for the model endpoint. There are pre-defined metrics for
                                 model endpoints such as predictions_per_second and latency_avg_5m but also custom
                                 metrics defined by the user. Please note that these metrics are stored in the time
                                 series DB and the results will be appeared under model_endpoint.spec.metrics.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A ModelEndpoint object.
        """

        # Parse JSON values into a dictionary
        feature_names = self._json_loads_if_not_none(endpoint.get("feature_names"))
        label_names = self._json_loads_if_not_none(endpoint.get("label_names"))
        feature_stats = self._json_loads_if_not_none(endpoint.get("feature_stats"))
        current_stats = self._json_loads_if_not_none(endpoint.get("current_stats"))
        children = self._json_loads_if_not_none(endpoint.get("children"))
        monitor_configuration = self._json_loads_if_not_none(
            endpoint.get("monitor_configuration")
        )
        endpoint_type = self._json_loads_if_not_none(endpoint.get("endpoint_type"))
        children_uids = self._json_loads_if_not_none(endpoint.get("children_uids"))
        labels = self._json_loads_if_not_none(endpoint.get("labels"))

        # Convert into model endpoint object
        endpoint_obj = mlrun.api.schemas.ModelEndpoint(
            metadata=mlrun.api.schemas.ModelEndpointMetadata(
                project=endpoint.get("project"),
                labels=labels,
                uid=endpoint.get("endpoint_id"),
            ),
            spec=mlrun.api.schemas.ModelEndpointSpec(
                function_uri=endpoint.get("function_uri"),
                model=endpoint.get("model"),
                model_class=endpoint.get("model_class"),
                model_uri=endpoint.get("model_uri"),
                feature_names=feature_names or None,
                label_names=label_names or None,
                stream_path=endpoint.get("stream_path"),
                algorithm=endpoint.get("algorithm"),
                monitor_configuration=monitor_configuration or None,
                active=endpoint.get("active"),
                monitoring_mode=endpoint.get("monitoring_mode"),
            ),
            status=mlrun.api.schemas.ModelEndpointStatus(
                state=endpoint.get("state") or None,
                feature_stats=feature_stats or None,
                current_stats=current_stats or None,
                children=children or None,
                first_request=endpoint.get("first_request"),
                last_request=endpoint.get("last_request"),
                accuracy=endpoint.get("accuracy"),
                error_count=endpoint.get("error_count"),
                drift_status=endpoint.get("drift_status"),
                endpoint_type=endpoint_type or None,
                children_uids=children_uids or None,
                monitoring_feature_set_uri=endpoint.get("monitoring_feature_set_uri")
                or None,
            ),
        )

        # If feature analysis was applied, add feature stats and current stats to the model endpoint result
        if feature_analysis and feature_names:
            endpoint_features = self.get_endpoint_features(
                feature_names=feature_names,
                feature_stats=feature_stats,
                current_stats=current_stats,
            )
            if endpoint_features:
                endpoint_obj.status.features = endpoint_features
                # Add the latest drift measures results (calculated by the model monitoring batch)
                drift_measures = self._json_loads_if_not_none(
                    endpoint.get("drift_measures")
                )
                endpoint_obj.status.drift_measures = drift_measures

        # If time metrics were provided, retrieve the results from the time series DB
        if metrics:
            endpoint_metrics = self.get_endpoint_metrics(
                endpoint_id=endpoint_obj.metadata.uid,
                start=start,
                end=end,
                metrics=metrics,
            )
            if endpoint_metrics:
                endpoint_obj.status.metrics = endpoint_metrics

        return endpoint_obj

    def _get_path_and_container(self):
        """Getting path and container based on the model monitoring configurations"""
        path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
            project=self.project,
            kind=mlrun.api.schemas.ModelMonitoringStoreKinds.ENDPOINTS,
        )
        (
            _,
            container,
            path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(path)
        return path, container

    def list_model_endpoints(
        self, model: str, function: str, labels: typing.List, top_level: bool
    ):
        """
        Returns a list of endpoint unique ids, supports filtering by model, function,
        labels or top level. By default, when no filters are applied, all available endpoint ids for the given project
        will be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key==value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.

        :return: List of model endpoints unique ids.
        """

        # Retrieve the raw data from the KV table and get the endpoint ids
        cursor = self.client.kv.new_cursor(
            container=self.container,
            table_path=self.path,
            filter_expression=self.build_kv_cursor_filter_expression(
                self.project,
                function,
                model,
                labels,
                top_level,
            ),
            attribute_names=["endpoint_id"],
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        )
        try:
            items = cursor.all()
        except Exception:
            return []

        # Create a list of model endpoints unique ids
        uids = [item["endpoint_id"] for item in items]

        return uids

    def delete_model_endpoints_resources(
        self, endpoints: mlrun.api.schemas.model_endpoints.ModelEndpointList
    ):
        """
        Delete all model endpoints resources in both KV and the time series DB.

        :param endpoints: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
        """

        # Delete model endpoint record from KV table
        for endpoint in endpoints.endpoints:
            self.delete_model_endpoint(
                endpoint.metadata.uid,
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

        # Cleanup TSDB
        frames = mlrun.utils.v3io_clients.get_frames_client(
            token=self.access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=self.container,
        )

        # Generate the required tsdb paths
        tsdb_path, filtered_path = self._generate_tsdb_paths()

        # Delete time series DB resources
        try:
            frames.delete(
                backend=model_monitoring_constants.StoreTarget.TSDB,
                table=filtered_path,
                if_missing=v3io_frames.frames_pb2.IGNORE,
            )
        except v3io_frames.errors.CreateError:
            # Frames might raise an exception if schema file does not exist.
            pass

        # Final cleanup of tsdb path
        tsdb_path.replace("://u", ":///u")
        store, _ = mlrun.store_manager.get_or_create_store(tsdb_path)
        store.rm(tsdb_path, recursive=True)

    def _generate_tsdb_paths(self) -> typing.Tuple[str, str]:
        """Generate a short path to the TSDB resources and a filtered path for the frames object

        :return: A tuple of:
             [0] = Short path to the TSDB resources
             [1] = Filtered path to TSDB events without schema and container
        """
        # Full path for the time series DB events
        full_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )

        # Generate the main directory with the TSDB resources
        tsdb_path = mlrun.utils.model_monitoring.parse_model_endpoint_project_prefix(
            full_path, self.project
        )

        # Generate filtered path without schema and container as required by the frames object
        (
            _,
            _,
            filtered_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(full_path)
        return tsdb_path, filtered_path

    @staticmethod
    def build_kv_cursor_filter_expression(
        project: str,
        function: str = None,
        model: str = None,
        labels: typing.List[str] = None,
        top_level: bool = False,
    ) -> str:
        """
        Convert the provided filters into a valid filter expression. The expected filter expression includes different
        conditions, divided by ' AND '.

        :param project:    The name of the project.
        :param model:      The name of the model to filter by.
        :param function:   The name of the function to filter by.
        :param labels:     A list of labels to filter by. Label filters work by either filtering a specific value of
                           a label (i.e. list("key==value")) or by looking for the existence of a given
                           key (i.e. "key").
        :param top_level:  If True will return only routers and endpoint that are NOT children of any router.

        :return: A valid filter expression as a string.
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
                f"(endpoint_type=='{str(mlrun.utils.model_monitoring.EndpointType.NODE_EP.value)}' "
                f"OR  endpoint_type=='{str(mlrun.utils.model_monitoring.EndpointType.ROUTER.value)}')"
            )

        return " AND ".join(filter_expression)

    @staticmethod
    def flatten_model_endpoint_attributes(
        endpoint: mlrun.api.schemas.ModelEndpoint,
    ) -> typing.Dict:
        """
        Retrieving flatten structure of the model endpoint object.

        :param endpoint: ModelEndpoint object that will be used for getting the attributes.

        :return: A flat dictionary of attributes.
        """

        # Prepare the data for the attributes dictionary
        labels = endpoint.metadata.labels or {}
        searchable_labels = {f"_{k}": v for k, v in labels.items()}
        feature_names = endpoint.spec.feature_names or []
        label_names = endpoint.spec.label_names or []
        feature_stats = endpoint.status.feature_stats or {}
        current_stats = endpoint.status.current_stats or {}
        children = endpoint.status.children or []
        endpoint_type = endpoint.status.endpoint_type or None
        children_uids = endpoint.status.children_uids or []

        # Fill the data. Note that because it is a flat dictionary, we use json.dumps() for encoding hierarchies
        # such as current_stats or label_names
        attributes = {
            "endpoint_id": endpoint.metadata.uid,
            "project": endpoint.metadata.project,
            "function_uri": endpoint.spec.function_uri,
            "model": endpoint.spec.model,
            "model_class": endpoint.spec.model_class or "",
            "labels": json.dumps(labels),
            "model_uri": endpoint.spec.model_uri or "",
            "stream_path": endpoint.spec.stream_path or "",
            "active": endpoint.spec.active or "",
            "monitoring_feature_set_uri": endpoint.status.monitoring_feature_set_uri
            or "",
            "monitoring_mode": endpoint.spec.monitoring_mode or "",
            "state": endpoint.status.state or "",
            "feature_stats": json.dumps(feature_stats),
            "current_stats": json.dumps(current_stats),
            "feature_names": json.dumps(feature_names),
            "children": json.dumps(children),
            "label_names": json.dumps(label_names),
            "endpoint_type": json.dumps(endpoint_type),
            "children_uids": json.dumps(children_uids),
            **searchable_labels,
        }
        return attributes

    @staticmethod
    def _json_loads_if_not_none(field: typing.Any) -> typing.Any:
        return json.loads(field) if field is not None else None

    @staticmethod
    def get_endpoint_features(
        feature_names: typing.List[str],
        feature_stats: dict = None,
        current_stats: dict = None,
    ) -> typing.List[mlrun.api.schemas.Features]:
        """
        Getting a new list of features that exist in feature_names along with their expected (feature_stats) and
        actual (current_stats) stats. The expected stats were calculated during the creation of the model endpoint,
        usually based on the data from the Model Artifact. The actual stats are based on the results from the latest
        model monitoring batch job.

        param feature_names: List of feature names.
        param feature_stats: Dictionary of feature stats that were stored during the creation of the model endpoint
                             object.
        param current_stats: Dictionary of the latest stats that were stored during the last run of the model monitoring
                             batch job.

        return: List of feature objects. Each feature has a name, weight, expected values, and actual values. More info
                can be found under mlrun.api.schemas.Features.
        """

        # Initialize feature and current stats dictionaries
        safe_feature_stats = feature_stats or {}
        safe_current_stats = current_stats or {}

        # Create feature object and add it to a general features list
        features = []
        for name in feature_names:
            if feature_stats is not None and name not in feature_stats:
                logger.warn("Feature missing from 'feature_stats'", name=name)
            if current_stats is not None and name not in current_stats:
                logger.warn("Feature missing from 'current_stats'", name=name)
            f = mlrun.api.schemas.Features.new(
                name, safe_feature_stats.get(name), safe_current_stats.get(name)
            )
            features.append(f)
        return features

    def get_endpoint_metrics(
        self,
        endpoint_id: str,
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
    ) -> typing.Dict[str, mlrun.api.schemas.Metric]:
        """
        Getting metrics from the time series DB. There are pre-defined metrics for model endpoints such as
        predictions_per_second and latency_avg_5m but also custom metrics defined by the user.

        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.

        :return: A dictionary of metrics in which the key is a metric name and the value is a Metric object that also
                 includes the relevant timestamp. More details about the Metric object can be found under
                 mlrun.api.schemas.Metric.
        """

        if not metrics:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Metric names must be provided"
            )

        # Initialize metrics mapping dictionary
        metrics_mapping = {}

        # Getting the path for the time series DB
        events_path = (
            mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
                project=self.project,
                kind=mlrun.api.schemas.ModelMonitoringStoreKinds.EVENTS,
            )
        )
        (
            _,
            _,
            events_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(events_path)

        # Retrieve the raw data from the time series DB based on the provided metrics and time ranges
        frames_client = mlrun.utils.v3io_clients.get_frames_client(
            token=self.access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=self.container,
        )

        try:
            data = frames_client.read(
                backend=model_monitoring_constants.StoreTarget.TSDB,
                table=events_path,
                columns=["endpoint_id", *metrics],
                filter=f"endpoint_id=='{endpoint_id}'",
                start=start,
                end=end,
            )

            # Fill the metrics mapping dictionary with the metric name and values
            data_dict = data.to_dict()
            for metric in metrics:
                metric_data = data_dict.get(metric)
                if metric_data is None:
                    continue

                values = [
                    (str(timestamp), value) for timestamp, value in metric_data.items()
                ]
                metrics_mapping[metric] = mlrun.api.schemas.Metric(
                    name=metric, values=values
                )
        except v3io_frames.errors.ReadError:
            logger.warn("Failed to read tsdb", endpoint=endpoint_id)
        return metrics_mapping


class _ModelEndpointSQLStore(_ModelEndpointStore):
    def write_model_endpoint(self, endpoint, update=True):
        raise NotImplementedError

    def update_model_endpoint(self, endpoint_id, attributes):
        raise NotImplementedError

    def delete_model_endpoint(self, endpoint_id):
        raise NotImplementedError

    def delete_model_endpoints_resources(
        self, endpoints: mlrun.api.schemas.model_endpoints.ModelEndpointList
    ):
        raise NotImplementedError

    def get_model_endpoint(
        self,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
        endpoint_id: str = None,
    ):
        raise NotImplementedError

    def list_model_endpoints(
        self, model: str, function: str, labels: typing.List, top_level: bool
    ):
        raise NotImplementedError


class ModelEndpointStoreType(enum.Enum):
    """Enum class to handle the different store type values for saving a model endpoint record."""

    kv = "kv"
    sql = "sql"

    def to_endpoint_target(
        self, project: str, access_key: str = None
    ) -> _ModelEndpointStore:
        """
        Return a ModelEndpointStore object based on the provided enum value.

        :param project:    The name of the project.
        :param access_key: Access key with permission to the DB table. Note that if access key is None and the
                           endpoint target is from type KV then the access key will be retrieved from the environment
                           variable.

        :return: ModelEndpointStore object.

        """

        if self.value == ModelEndpointStoreType.kv.value:

            # Get V3IO access key from env
            access_key = (
                mlrun.mlconf.get_v3io_access_key() if access_key is None else access_key
            )

            return _ModelEndpointKVStore(project=project, access_key=access_key)

        # Assuming SQL store target if store type is not KV.
        # Update these lines once there are more than two store target types.
        return _ModelEndpointSQLStore(project=project)

    @classmethod
    def _missing_(cls, value: typing.Any):
        """A lookup function to handle an invalid value.
        :param value: Provided enum (invalid) value.
        """
        valid_values = list(cls.__members__.keys())
        raise mlrun.errors.MLRunInvalidArgumentError(
            "%r is not a valid %s, please choose a valid value: %s."
            % (value, cls.__name__, valid_values)
        )


def get_model_endpoint_target(
    project: str, access_key: str = None
) -> _ModelEndpointStore:
    """
    Getting the DB target type based on mlrun.config.model_endpoint_monitoring.store_type.

    :param project:    The name of the project.
    :param access_key: Access key with permission to the DB table.

    :return: ModelEndpointStore object. Using this object, the user can apply different operations on the
             model endpoint record such as write, update, get and delete.
    """

    # Get store type value from ModelEndpointStoreType enum class
    model_endpoint_store_type = ModelEndpointStoreType(
        mlrun.mlconf.model_endpoint_monitoring.store_type
    )

    # Convert into model endpoint store target object
    return model_endpoint_store_type.to_endpoint_target(project, access_key)
