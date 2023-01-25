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

import mlrun
import mlrun.api.schemas
import mlrun.model_monitoring.constants as model_monitoring_constants
from mlrun.utils import logger


class ModelEndpointStore(ABC):
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

        :param endpoint: `ModelEndpoint` object that will be written into the DB.
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

        :param endpoints: An object of `ModelEndpointList` which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
        """
        pass

    @abstractmethod
    def get_model_endpoint(
        self,
        endpoint_id: str,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        feature_analysis: bool = False,
        convert_to_endpoint_object: bool = True,
    ) -> typing.Union[mlrun.api.schemas.ModelEndpoint, dict]:
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
        :param metrics:                    A list of real-time metrics to return for the model endpoint. There are
                                           pre-defined real-time metrics for model endpoints such as
                                           `predictions_per_second` and `latency_avg_5m` but also custom metrics defined
                                           by the user. Please note that these metrics are stored in the time series
                                           DB and the results will be appeared under `model_endpoint.spec.metrics`.
        :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                           be added to the output of the resulting object.
        :param convert_to_endpoint_object: A boolean that indicates whether to convert the model endpoint dictionary
                                           into a `ModelEndpoint` or not. True by default.

        :return: A `ModelEndpoint` object or a model endpoint dictionary if `convert_to_endpoint_object` is False.
        """
        pass

    @abstractmethod
    def list_model_endpoints(
        self,
        model: str = None,
        function: str = None,
        labels: typing.List[str] = None,
        top_level: bool = None,
        metrics: typing.List[str] = None,
        start: str = "now-1h",
        end: str = "now",
        uids: typing.List = None,
    ) -> mlrun.api.schemas.ModelEndpointList:
        """
        Returns a list of `ModelEndpoint` objects, supports filtering by model, function, labels or top level.
        By default, when no filters are applied, all available `ModelEndpoint` objects for the given project will
        be listed.

        :param model:           The name of the model to filter by.
        :param function:        The name of the function to filter by.
        :param labels:          A list of labels to filter by. Label filters work by either filtering a specific value
                                of a label (i.e. list("key=value")) or by looking for the existence of a given
                                key (i.e. "key").
        :param top_level:       If True will return only routers and endpoint that are NOT children of any router.
        :param metrics:         A list of real-time metrics to return for each model endpoint. There are pre-defined
                                real-time metrics for model endpoints such as `predictions_per_second` and
                                `latency_avg_5m` but also custom metrics defined by the user. Please note that these
                                metrics are stored in the time series DB and the results will be appeared under
                                `model_endpoint.spec.metrics`.
        :param start:           The start time of the metrics. Can be represented by a string containing an RFC 3339
                                time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for
                                 the earliest time.
        :param uids:             List of model endpoint unique ids to include in the result.

        :return: An object of `ModelEndpointList` which is literally a list of model endpoints along with some
                          metadata. To get a standard list of model endpoints use `ModelEndpointList.endpoints`.
        """
        pass

    @staticmethod
    def get_params(endpoint: mlrun.api.schemas.ModelEndpoint) -> typing.Dict:
        """
        Retrieving the relevant attributes from the model endpoint object.

        :param endpoint: `ModelEndpoint` object that will be used for getting the attributes.

        :return: A flat dictionary of attributes.
        """
        return endpoint.flat_dict()

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
                can be found under `mlrun.api.schemas.Features`.
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
        Create a `ModelEndpoint` object according to a provided model endpoint dictionary.

        :param endpoint:         DB record of model endpoint which need to be converted into a valid `ModelEndpoint`
                                 object.
        :param feature_analysis: When True, the base feature statistics and current feature statistics will be added to
                                 the output of the resulting object.

        :return: A `ModelEndpoint` object.
        """

        endpoint_obj = mlrun.api.schemas.ModelEndpoint.from_dict(endpoint)

        # If feature analysis was applied, add feature stats and current stats to the model endpoint result
        if feature_analysis and endpoint_obj.spec.feature_names:

            endpoint_features = self.get_endpoint_features(
                feature_names=endpoint_obj.spec.feature_names,
                feature_stats=endpoint_obj.status.feature_stats,
                current_stats=endpoint_obj.status.current_stats,
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

    @abstractmethod
    def get_endpoint_real_time_metrics(
        self,
        endpoint_id: str,
        metrics: typing.List[str],
        start: str = "now-1h",
        end: str = "now",
        access_key: str = None,
    ) -> typing.Dict[str, typing.List]:
        """
        Getting metrics from the time series DB. There are pre-defined metrics for model endpoints such as
        `predictions_per_second` and `latency_avg_5m` but also custom metrics defined by the user.

        :param endpoint_id:      The unique id of the model endpoint.
        :param metrics:          A list of real-time metrics to return for the model endpoint.
        :param start:            The start time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param end:              The end time of the metrics. Can be represented by a string containing an RFC 3339
                                 time, a Unix timestamp in milliseconds, a relative time (`'now'` or
                                 `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, and `'d'` = days), or 0 for the
                                 earliest time.
        :param access_key:       V3IO access key that will be used for generating Frames client object. If not
                                 provided, the access key will be retrieved from the environment variables.

        :return: A dictionary of metrics in which the key is a metric name and the value is a list of tuples that
                 includes timestamps and the values.
        """

        pass
