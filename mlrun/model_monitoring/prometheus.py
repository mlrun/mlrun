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
#

import prometheus_client

from mlrun.common.schemas.model_monitoring import EventFieldType, PrometheusMetric

# Memory path for Prometheus registry file
_registry_path = "/tmp/prom-reg.txt"

# Initializing Promethues metric collector registry
_registry: prometheus_client.CollectorRegistry = prometheus_client.CollectorRegistry()

# The following real-time metrics are being updated through the monitoring stream graph steps
_prediction_counter: prometheus_client.Counter = prometheus_client.Counter(
    name=PrometheusMetric.PREDICTIONS_TOTAL,
    documentation="Counter for total predictions",
    registry=_registry,
    labelnames=[
        EventFieldType.PROJECT,
        EventFieldType.ENDPOINT_ID,
        EventFieldType.MODEL,
        EventFieldType.ENDPOINT_TYPE,
    ],
)
_model_latency: prometheus_client.Summary = prometheus_client.Summary(
    name=PrometheusMetric.MODEL_LATENCY_SECONDS,
    documentation="Summary for for model latency",
    registry=_registry,
    labelnames=[
        EventFieldType.PROJECT,
        EventFieldType.ENDPOINT_ID,
        EventFieldType.MODEL,
        EventFieldType.ENDPOINT_TYPE,
    ],
)
_income_features: prometheus_client.Gauge = prometheus_client.Gauge(
    name=PrometheusMetric.INCOME_FEATURES,
    documentation="Samples of features and predictions",
    registry=_registry,
    labelnames=[
        EventFieldType.PROJECT,
        EventFieldType.ENDPOINT_ID,
        EventFieldType.METRIC,
    ],
)
_error_counter: prometheus_client.Counter = prometheus_client.Counter(
    name=PrometheusMetric.ERRORS_TOTAL,
    documentation="Counter for total errors",
    registry=_registry,
    labelnames=[
        EventFieldType.PROJECT,
        EventFieldType.ENDPOINT_ID,
        EventFieldType.MODEL,
    ],
)

# The following metrics are being updated through the model monitoring batch job
_batch_metrics: prometheus_client.Gauge = prometheus_client.Gauge(
    name=PrometheusMetric.DRIFT_METRICS,
    documentation="Results from the batch drift analysis",
    registry=_registry,
    labelnames=[
        EventFieldType.PROJECT,
        EventFieldType.ENDPOINT_ID,
        EventFieldType.METRIC,
    ],
)
_drift_status: prometheus_client.Enum = prometheus_client.Enum(
    name=PrometheusMetric.DRIFT_STATUS,
    documentation="Drift status of the model endpoint",
    registry=_registry,
    states=["NO_DRIFT", "DRIFT_DETECTED", "POSSIBLE_DRIFT"],
    labelnames=[EventFieldType.PROJECT, EventFieldType.ENDPOINT_ID],
)


def _write_registry(func):
    def wrapper(*args, **kwargs):
        global _registry
        """A wrapper function to update the registry file each time a metric has been updated"""
        func(*args, **kwargs)
        prometheus_client.write_to_textfile(path=_registry_path, registry=_registry)

    return wrapper


@_write_registry
def write_predictions_and_latency_metrics(
    project: str, endpoint_id: str, latency: int, model_name: str, endpoint_type: int
):
    """
    Update the prediction counter and the latency value of the provided model endpoint within Prometheus registry.
    Please note that while the prediction counter is ALWAYS increasing by 1,the latency summary metric is being
    increased by the event latency time. Grafana dashboard will query the average latency time by dividing the total
    latency value by the total amount of predictions.

    :param project:       Project name.
    :param endpoint_id:   Model endpoint unique id.
    :param latency:       Latency time (microsecond) in which the event has been processed through the model server.
    :param model_name:    Model name which will be used by Grafana for displaying the results by model.
    :param endpoint_type: Endpoint type that is represented by an int (possible values: 1,2,3) corresponding to the
                          Enum class :py:class:`~mlrun.common.schemas.model_monitoring.EndpointType`.
    """

    # Increase the prediction counter by 1
    _prediction_counter.labels(
        project=project,
        endpoint_id=endpoint_id,
        model=model_name,
        endpoint_type=endpoint_type,
    ).inc(1)

    # Increase the latency value according to the provided latency of the current event
    _model_latency.labels(
        project=project,
        endpoint_id=endpoint_id,
        model=model_name,
        endpoint_type=endpoint_type,
    ).observe(latency)


@_write_registry
def write_income_features(project: str, endpoint_id: str, features: dict[str, float]):
    """Update a sample of features.

    :param project:     Project name.
    :param endpoint_id: Model endpoint unique id.
    :param features:    Dictionary in which the key is a feature name and the value is a float number.


    """

    for metric in features:
        _income_features.labels(
            project=project, endpoint_id=endpoint_id, metric=metric
        ).set(value=features[metric])


@_write_registry
def write_drift_metrics(project: str, endpoint_id: str, metric: str, value: float):
    """Update drift metrics that have been calculated through the monitoring batch job

    :param project:     Project name.
    :param endpoint_id: Model endpoint unique id.
    :param metric:      Metric name (e.g. TVD, Hellinger).
    :param value:       Metric value as a float.

    """

    _batch_metrics.labels(project=project, endpoint_id=endpoint_id, metric=metric).set(
        value=value
    )


@_write_registry
def write_drift_status(project: str, endpoint_id: str, drift_status: str):
    """
    Update the drift status enum for a specific model endpoint.

    :param project:      Project name.
    :param endpoint_id:  Model endpoint unique id.
    :param drift_status: Drift status value, can be one of the following: 'NO_DRIFT', 'DRIFT_DETECTED', or
                         'POSSIBLE_DRIFT'.
    """

    _drift_status.labels(project=project, endpoint_id=endpoint_id).state(drift_status)


@_write_registry
def write_errors(project: str, endpoint_id: str, model_name: str):
    """
    Update the error counter for a specific model endpoint.

    :param project:     Project name.
    :param endpoint_id: Model endpoint unique id.
    :param model_name:  Model name. Will be used by Grafana to show the amount of errors per model by time.
    """

    _error_counter.labels(
        project=project, endpoint_id=endpoint_id, model=model_name
    ).inc(1)


def get_registry() -> str:
    """Returns the parsed registry file according to the exposition format of Prometheus."""

    # Read the registry file (note that the text is stored in UTF-8 format)
    f = open(_registry_path)
    lines = f.read()
    f.close()

    # Reset part of the metrics to avoid a repeating scraping of the same value
    clean_metrics()

    return lines


@_write_registry
def clean_metrics():
    """Clean the income features values. As these results are relevant only for a certain timestamp, we will remove
    them from the global registry after they have been scraped by Prometheus."""

    _income_features.clear()
