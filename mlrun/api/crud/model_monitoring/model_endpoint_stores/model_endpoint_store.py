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

import json
import typing
from abc import ABC, abstractmethod

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
                           of the attributes dictionary should exist in the DB table.

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
        convert_to_endpoint_object: bool = True,
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Get a single model endpoint object. You can apply different time series metrics that will be added to the
        result.

        :param endpoint_id:                The unique id of the model endpoint.
        :param start:                      The start time of the metrics. Can be represented by a string containing
                                           an RFC 3339 time, a Unix timestamp in milliseconds, a relative time
                                           (`'now'` or `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and
                                           `'d'` = days), or 0 for the earliest time.
        :param end:                        The end time of the metrics. Can be represented by a string containing an
                                           RFC 3339 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                           `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days),
                                           or 0 for the earliest time.
        :param metrics:                    A list of metrics to return for the model endpoint. There are pre-defined
                                           metrics for model endpoints such as predictions_per_second and
                                           latency_avg_5m but also custom metrics defined by the user. Please note
                                           that these metrics are stored in the time series DB and the results will
                                           be appeared under model_endpoint.spec.metrics.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.
        :param convert_to_endpoint_object: A boolean that indicates whether to convert the model endpoint dictionary
                                           into a ModelEndpoint or not. True by default.

        :return: A ModelEndpoint object.
        """
        pass

    @abstractmethod
    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: typing.List = None,
        top_level: bool = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        uids: typing.List = None,
    ) -> mlrun.api.schemas.ModelEndpointList:
        """
        Returns a list of ModelEndpoint objects, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available ModelEndpoint objects for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key==value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param metrics:         A list of metrics to return for each model endpoint. There are pre-defined metrics
                                for model endpoints such as predictions_per_second and latency_avg_5m but also custom
                                metrics defined by the user. Please note that these metrics are stored in the time
                                series DB and the results will be appeared under model_endpoint.spec.metrics.
        :param start:           The start time of the metrics. Can be represented by a string containing an RFC 3339
                                time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for
                                 the earliest time.
        :param uids:             List of model endpoint unique ids to include in the result.

        :return: An object of ModelEndpointList which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use ModelEndpointList.endpoints.
        """
        pass

    @staticmethod
    def get_params(endpoint: mlrun.api.schemas.ModelEndpoint) -> typing.Dict:
        """
        Retrieving the relevant attributes from the model endpoint object.

        :param endpoint: ModelEndpoint object that will be used for getting the attributes.

        :return: A flat dictionary of attributes.
        """

        # Prepare the data for the attributes dictionary
        labels = endpoint.metadata.labels or {}
        feature_names = endpoint.spec.feature_names or []
        label_names = endpoint.spec.label_names or []
        feature_stats = endpoint.status.feature_stats or {}
        current_stats = endpoint.status.current_stats or {}
        children = endpoint.status.children or []
        endpoint_type = endpoint.status.endpoint_type or None
        children_uids = endpoint.status.children_uids or []
        predictions_per_second = endpoint.status.predictions_per_second or None
        latency_avg_1h = endpoint.status.latency_avg_1h or None

        # Fill the data. Note that because it is a flat dictionary, we use json.dumps() for encoding hierarchies
        # such as current_stats or label_names
        attributes = {
            model_monitoring_constants.EventFieldType.ENDPOINT_ID: endpoint.metadata.uid,
            model_monitoring_constants.EventFieldType.PROJECT: endpoint.metadata.project,
            model_monitoring_constants.EventFieldType.FUNCTION_URI: endpoint.spec.function_uri,
            model_monitoring_constants.EventFieldType.MODEL: endpoint.spec.model,
            model_monitoring_constants.EventFieldType.MODEL_CLASS: endpoint.spec.model_class
            or "",
            model_monitoring_constants.EventFieldType.LABELS: json.dumps(labels),
            model_monitoring_constants.EventFieldType.MODEL_URI: endpoint.spec.model_uri
            or "",
            model_monitoring_constants.EventFieldType.STREAM_PATH: endpoint.spec.stream_path
            or "",
            model_monitoring_constants.EventFieldType.ACTIVE: endpoint.spec.active
            or "",
            model_monitoring_constants.EventFieldType.FEATURE_SET_URI: endpoint.status.monitoring_feature_set_uri
            or "",
            model_monitoring_constants.EventFieldType.MONITORING_MODE: endpoint.spec.monitoring_mode
            or "",
            model_monitoring_constants.EventFieldType.STATE: endpoint.status.state
            or "",
            model_monitoring_constants.EventFieldType.FEATURE_STATS: json.dumps(
                feature_stats
            ),
            model_monitoring_constants.EventFieldType.CURRENT_STATS: json.dumps(
                current_stats
            ),
            model_monitoring_constants.EventLiveStats.PREDICTIONS_PER_SECOND: json.dumps(
                predictions_per_second
            )
            if predictions_per_second is not None
            else None,
            model_monitoring_constants.EventLiveStats.LATENCY_AVG_1H: json.dumps(
                latency_avg_1h
            )
            if latency_avg_1h is not None
            else None,
            model_monitoring_constants.EventFieldType.FEATURE_NAMES: json.dumps(
                feature_names
            ),
            model_monitoring_constants.EventFieldType.CHILDREN: json.dumps(children),
            model_monitoring_constants.EventFieldType.LABEL_NAMES: json.dumps(
                label_names
            ),
            model_monitoring_constants.EventFieldType.ENDPOINT_TYPE: json.dumps(
                endpoint_type
            ),
            model_monitoring_constants.EventFieldType.CHILDREN_UIDS: json.dumps(
                children_uids
            ),
        }
        return attributes

    @staticmethod
    def _json_loads_if_not_none(field: typing.Any) -> typing.Any:
        return (
            json.loads(field)
            if field and field != "null" and field is not None
            else None
        )

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

    def _convert_into_model_endpoint_object(
        self, endpoint: typing.Dict, feature_analysis: bool = False
    ) -> mlrun.api.schemas.ModelEndpoint:
        """
        Create a ModelEndpoint object according to a provided model endpoint dictionary.

        :param endpoint:         DB record of model endpoint which need to be converted into a valid ModelEndpoint
                                 object.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A ModelEndpoint object.
        """

        # Parse JSON values into a dictionary
        feature_names = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.FEATURE_NAMES)
        )
        label_names = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.LABEL_NAMES)
        )
        feature_stats = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.FEATURE_STATS)
        )
        current_stats = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.CURRENT_STATS)
        )
        children = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.CHILDREN)
        )
        monitor_configuration = self._json_loads_if_not_none(
            endpoint.get(
                model_monitoring_constants.EventFieldType.MONITOR_CONFIGURATION
            )
        )
        endpoint_type = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.ENDPOINT_TYPE)
        )
        children_uids = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.CHILDREN_UIDS)
        )
        labels = self._json_loads_if_not_none(
            endpoint.get(model_monitoring_constants.EventFieldType.LABELS)
        )

        # Convert into model endpoint object
        endpoint_obj = mlrun.api.schemas.ModelEndpoint(
            metadata=mlrun.api.schemas.ModelEndpointMetadata(
                project=endpoint.get(model_monitoring_constants.EventFieldType.PROJECT),
                labels=labels,
                uid=endpoint.get(model_monitoring_constants.EventFieldType.ENDPOINT_ID),
            ),
            spec=mlrun.api.schemas.ModelEndpointSpec(
                function_uri=endpoint.get(
                    model_monitoring_constants.EventFieldType.FUNCTION_URI
                ),
                model=endpoint.get(model_monitoring_constants.EventFieldType.MODEL),
                model_class=endpoint.get(
                    model_monitoring_constants.EventFieldType.MODEL_CLASS
                ),
                model_uri=endpoint.get(
                    model_monitoring_constants.EventFieldType.MODEL_URI
                ),
                feature_names=feature_names or None,
                label_names=label_names or None,
                stream_path=endpoint.get(
                    model_monitoring_constants.EventFieldType.STREAM_PATH
                ),
                algorithm=endpoint.get(
                    model_monitoring_constants.EventFieldType.ALGORITHM
                ),
                monitor_configuration=monitor_configuration or None,
                active=endpoint.get(model_monitoring_constants.EventFieldType.ACTIVE),
                monitoring_mode=endpoint.get(
                    model_monitoring_constants.EventFieldType.MONITORING_MODE
                ),
            ),
            status=mlrun.api.schemas.ModelEndpointStatus(
                state=endpoint.get(model_monitoring_constants.EventFieldType.STATE)
                or None,
                feature_stats=feature_stats or None,
                current_stats=current_stats or None,
                children=children or None,
                first_request=endpoint.get(
                    model_monitoring_constants.EventFieldType.FIRST_REQUEST
                ),
                last_request=endpoint.get(
                    model_monitoring_constants.EventFieldType.LAST_REQUEST
                ),
                accuracy=endpoint.get(
                    model_monitoring_constants.EventFieldType.ACCURACY
                ),
                error_count=endpoint.get(
                    model_monitoring_constants.EventFieldType.ERROR_COUNT
                ),
                drift_status=endpoint.get(
                    model_monitoring_constants.EventFieldType.DRIFT_STATUS
                ),
                endpoint_type=endpoint_type or None,
                children_uids=children_uids or None,
                monitoring_feature_set_uri=endpoint.get(
                    model_monitoring_constants.EventFieldType.MONITORING_FEATURE_SET_URI
                )
                or None,
                predictions_per_second=endpoint.get(
                    model_monitoring_constants.EventLiveStats.PREDICTIONS_PER_SECOND
                )
                if endpoint.get("predictions_per_second") != "null"
                else None,
                latency_avg_1h=endpoint.get(
                    model_monitoring_constants.EventLiveStats.LATENCY_AVG_1H
                )
                if endpoint.get("latency_avg_1h") != "null"
                else None,
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
                    endpoint.get(
                        model_monitoring_constants.EventFieldType.DRIFT_MEASURES
                    )
                )
                endpoint_obj.status.drift_measures = drift_measures

        return endpoint_obj

    def get_endpoint_metrics(
        self,
        endpoint_id: str,
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = mlrun.mlconf.get_v3io_access_key(),
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
        :param access_key:       V3IO access key that will be used for generating Frames client object. By default,
                                 the access key will be retrieved from the environment variables.

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
            container,
            events_path,
        ) = mlrun.utils.model_monitoring.parse_model_endpoint_store_prefix(events_path)

        # Retrieve the raw data from the time series DB based on the provided metrics and time ranges
        frames_client = mlrun.utils.v3io_clients.get_frames_client(
            token=access_key,
            address=mlrun.mlconf.v3io_framesd,
            container=container,
        )

        try:
            data = frames_client.read(
                backend=model_monitoring_constants.TimeSeriesTarget.TSDB,
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
