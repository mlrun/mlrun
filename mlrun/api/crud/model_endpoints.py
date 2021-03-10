import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping

import pandas as pd
from v3io.dataplane import RaiseForStatus

from mlrun.api.schemas import Features, Metric, ModelEndpoint, ModelEndpointState
from mlrun.artifacts import get_model
from mlrun.config import config
from mlrun.errors import (
    MLRunBadRequestError,
    MLRunInvalidArgumentError,
    MLRunNotFoundError,
)
from mlrun.utils.helpers import logger
from mlrun.utils.v3io_clients import get_frames_client, get_v3io_client

ENDPOINTS_TABLE_PATH = "model-endpoints/endpoints"
ENDPOINT_EVENTS_TABLE_PATH = "model-endpoints/events"
ENDPOINT_TABLE_ATTRIBUTES = [
    "project",
    "model",
    "function",
    "tag",
    "model_class",
    "labels",
    "first_request",
    "last_request",
    "error_count",
    "drift_status",
    "base_stats",
    "current_stats",
    "drift_measurements",
]


@dataclass
class TimeMetric:
    tsdb_column: str
    metric_name: str
    headers: List[str]

    def transform_df_to_metric(self, data: pd.DataFrame) -> Optional[Metric]:
        if data.empty or self.tsdb_column not in data.columns:
            return None

        values = data[self.tsdb_column].reset_index().to_numpy()
        describe = data[self.tsdb_column].describe().to_dict()

        return Metric(
            name=self.metric_name,
            start_timestamp=str(data.index[0]),
            end_timestamp=str(data.index[-1]),
            headers=self.headers,
            values=[(str(timestamp), float(value)) for timestamp, value in values],
            min=describe["min"],
            avg=describe["mean"],
            max=describe["max"],
        )


class ModelEndpoints:
    @staticmethod
    def register_endpoint(
        access_key: str,
        project: str,
        model: str,
        function: str,
        tag: Optional[str] = None,
        model_class: Optional[str] = None,
        labels: Optional[dict] = None,
        model_artifact: Optional[str] = None,
        feature_stats: Optional[dict] = None,
        feature_names: Optional[List[str]] = None,
        monitor_configuration: Optional[dict] = None,
        stream_path: Optional[str] = None,
        active: bool = True,
        update: bool = False,
    ):
        """
        Writes endpoint data to KV, a prerequisite for initializing the monitoring process for a specific endpoint.
        This method exposes a functionality for storing endpoint meta data that is crucial for the endpoint monitoring
        system.

        :param access_key: V3IO access key for managing user permissions

        Parameters for ModelEndpointMetadata
        :param project: The name of the project of which this endpoint belongs to (used for creating endpoint.id)
        :param tag: The tag/version of the model/function (used for creating endpoint.id)
        :param labels: key value pairs of user defined labels
        :param model_artifact: The path to the model artifact containing metadata about the features of the model
        :param feature_stats: The actual metadata about the features of the model
        :param feature_names: A list of feature names, if provided along side `model_artifact` or `feature_stats`, will
        override the feature names that can be found in the metadata
        :param monitor_configuration: A monitoring related key value configuration
        :param stream_path: The path to the output stream of the model server

        Parameters for ModelEndpointSpec
        :param model: The name of the model that is used in the serving function (used for creating endpoint.id)
        :param function: The name of the function that servers the model (used for creating endpoint.id)
        :param model_class: The class of the model

        Parameters for ModelEndpointStatus
        :param active: The "activation" status of the endpoint - True for active / False for not active (default True)
        :param update: When False, if endpoint already exists, don't write endpoint's data again
        """

        # if endpoint already exists and 'update' is False, don't try to write it to kv again
        if not update:
            try:
                endpoint_id = ModelEndpoint.create_endpoint_id(
                    project=project, function=function, model=model, tag=tag
                )
                deserialize_endpoint_from_kv(
                    access_key=access_key, project=project, endpoint_id=endpoint_id
                )
                return
            except MLRunNotFoundError:
                pass

        if model_artifact or feature_stats:
            logger.info(
                "Getting feature metadata",
                project=project,
                model=model,
                function=function,
                tag=tag,
                model_artifact=model_artifact,
            )

        # If model artifact was supplied but `feature_stats` was not, grab model artifact and get `feature_stats`
        if model_artifact and not feature_stats:
            logger.info(
                "Getting model object, inferring column names and collecting feature stats"
            )
            if model_artifact:
                model_obj = get_model(model_artifact)
                feature_stats = model_obj[1].feature_stats

        # If `feature_stats` was either populated by `model_artifact`or by manual input, make sure to keep the names
        # of the features. If `feature_names` was supplied, replace the names set in `feature_stats`, otherwise - make
        # sure to keep a clean version of the names
        if feature_stats:
            logger.info("Feature stats found, cleaning feature names")
            if feature_names:
                if len(feature_stats) != len(feature_names):
                    raise MLRunInvalidArgumentError(
                        f"`feature_stats` and `feature_names` have a different number of names, while expected to match"
                        f"feature_stats({len(feature_stats)}), feature_names({len(feature_names)})"
                    )
            clean_feature_stats = {}
            clean_feature_names = []
            for i, (feature, stats) in enumerate(feature_stats.items()):
                if feature_names:
                    clean_name = _clean_feature_name(feature_names[i])
                else:
                    clean_name = _clean_feature_name(feature)
                clean_feature_stats[clean_name] = stats
                clean_feature_names.append(clean_name)
            feature_stats = clean_feature_stats
            feature_names = clean_feature_names

            logger.info(
                "Done preparing feature names and stats", feature_names=feature_names
            )

        # If none of the above was supplied, feature names will be assigned on first contact with the model monitoring
        # system

        endpoint = ModelEndpoint.new(
            project=project,
            model=model,
            function=function,
            tag=tag,
            model_class=model_class,
            labels=labels,
            model_artifact=model_artifact,
            stream_path=stream_path,
            feature_stats=feature_stats,
            feature_names=feature_names,
            monitor_configuration=monitor_configuration,
            active=active,
            state="registered",
        )

        logger.info("Registering model endpoint", endpoint_id=endpoint.id)

        serialize_endpoint_to_kv(access_key, endpoint, update)

        logger.info("Model endpoint registered", endpoint_id=endpoint.id)

        return endpoint

    @staticmethod
    def update_endpoint_record(
        access_key: str,
        project: str,
        endpoint_id: str,
        payload: dict,
        check_existence: bool = True,
    ):
        """
        Updates the KV data of a given model endpoint

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        :param payload: The parameters that should be updated
        :param check_existence: Check if the endpoint already exists, if it doesn't, don't create the record and raise
        MLRunInvalidArgumentError
        """

        if not payload:
            raise MLRunInvalidArgumentError(
                "Update payload must contain at least one field to update"
            )

        logger.info("Updating model endpoint table", endpoint_id=endpoint_id)
        client = get_v3io_client(endpoint=config.v3io_api)

        if check_existence:
            try:
                client.kv.get(
                    container=config.model_endpoint_monitoring.container,
                    table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
                    key=endpoint_id,
                    access_key=access_key,
                )
            except RuntimeError:
                raise MLRunInvalidArgumentError(f"Endpoint: {endpoint_id} not found")

        client.kv.update(
            container=config.model_endpoint_monitoring.container,
            table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
            key=endpoint_id,
            access_key=access_key,
            attributes=payload,
        )
        logger.info("Model endpoint table updated", endpoint_id=endpoint_id)

    @staticmethod
    def clear_endpoint_record(access_key: str, project: str, endpoint_id: str):
        """
        Clears the KV data of a given model endpoint

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        """

        logger.info("Clearing model endpoint table", endpoint_id=endpoint_id)
        client = get_v3io_client(endpoint=config.v3io_api)
        client.kv.delete(
            container=config.model_endpoint_monitoring.container,
            table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
            key=endpoint_id,
            access_key=access_key,
        )

        logger.info("Model endpoint table cleared", endpoint_id=endpoint_id)

    @staticmethod
    def list_endpoints(
        access_key: str,
        project: str,
        model: Optional[str] = None,
        function: Optional[str] = None,
        tag: Optional[str] = None,
        labels: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        start: str = "now-1h",
        end: str = "now",
    ) -> List[ModelEndpointState]:
        """
        Returns a list of `ModelEndpointState` objects. Each object represents the current state of a model endpoint.
        This functions supports filtering by the following parameters:
        1) model
        2) function
        3) tag
        4) labels
        By default, when no filters are applied, all available endpoints for the given project will be listed.

        In addition, this functions provides a facade for listing endpoint related metrics. This facade is time-based
        and depends on the 'start' and 'end' parameters. By default, when the metrics parameter is None, no metrics are
        added to the output of this function.

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param model: The name of the model to filter by
        :param function: The name of the function to filter by
        :param tag: A tag to filter by
        :param labels: A list of labels to filter by. Label filters work by either filtering a specific value of a label
        (i.e. list("key==value")) or by looking for the existence of a given key (i.e. "key")
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics
        :param end: The end time of the metrics
        """

        logger.info(
            "Listing endpoints",
            project=project,
            model=model,
            function=function,
            tag=tag,
            labels=labels,
            metrics=metrics,
            start=start,
            end=end,
        )

        client = get_v3io_client(endpoint=config.v3io_api)
        cursor = client.kv.new_cursor(
            container=config.model_endpoint_monitoring.container,
            table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
            access_key=access_key,
            filter_expression=build_kv_cursor_filter_expression(
                project, function, model, tag, labels
            ),
        )
        endpoints = cursor.all()

        endpoint_state_list = []
        for endpoint in endpoints:
            endpoint_metrics = {}
            if metrics:
                endpoint_metrics = get_endpoint_metrics(
                    access_key=access_key,
                    project=project,
                    endpoint_id=endpoint.get("id"),
                    name=metrics,
                    start=start,
                    end=end,
                )

            # Collect labels (by convention labels are labeled with underscore '_'), ignore builtin '__name' field
            state = ModelEndpointState(
                endpoint=ModelEndpoint.new(
                    project=endpoint.get("project"),
                    model=endpoint.get("model"),
                    function=endpoint.get("function"),
                    tag=endpoint.get("tag"),
                    model_class=endpoint.get("model_class"),
                    labels=json.loads(endpoint.get("labels")),
                    model_artifact=endpoint.get("model_artifact"),
                    stream_path=endpoint.get("stream_path"),
                    feature_stats=json.loads(endpoint.get("feature_stats")),
                    state=endpoint.get("state"),
                    active=endpoint.get("active"),
                ),
                first_request=endpoint.get("first_request"),
                last_request=endpoint.get("last_request"),
                error_count=endpoint.get("error_count"),
                drift_status=endpoint.get("drift_status"),
                metrics=endpoint_metrics,
            )
            endpoint_state_list.append(state)

        return endpoint_state_list

    @staticmethod
    def get_endpoint(
        access_key: str,
        project: str,
        endpoint_id: str,
        metrics: Optional[List[str]] = None,
        start: str = "now-1h",
        end: str = "now",
        features: bool = False,
    ) -> ModelEndpointState:
        """
        Returns the current state of an endpoint.


        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param endpoint_id: The id of the model endpoint
        :param metrics: A list of metrics to return for each endpoint, read more in 'TimeMetric'
        :param start: The start time of the metrics
        :param end: The end time of the metrics
        :param features: When True, the base feature statistics and current feature statistics will be added to the
        output of the resulting object
        """
        endpoint_state = deserialize_endpoint_state_from_kv(
            access_key=access_key, project=project, endpoint_id=endpoint_id
        )

        if not endpoint_state:
            url = f"/projects/{project}/model-endpoints/{endpoint_id}"
            raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found - {url}")

        if metrics:
            endpoint_metrics = get_endpoint_metrics(
                access_key=access_key,
                project=project,
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                name=metrics,
            )
            endpoint_state.metrics = endpoint_metrics

        if features:
            endpoint_features = get_endpoint_features(
                feature_names=endpoint_state.endpoint.metadata.feature_names,
                feature_stats=endpoint_state.endpoint.metadata.feature_stats,
                current_stats=endpoint_state.current_stats,
            )
            endpoint_state.features = endpoint_features

        return endpoint_state


def serialize_endpoint_to_kv(
    access_key: str, endpoint: ModelEndpoint, update: bool = True
):
    """
    Writes endpoint data to KV, a prerequisite for initializing the monitoring process

    :param access_key: V3IO access key for managing user permissions
    :param endpoint: ModelEndpoint object
    :param update: When True, use `client.kv.update`, otherwise use `client.kv.put`
    """

    labels = endpoint.metadata.labels or {}
    searchable_labels = {f"_{k}": v for k, v in labels.items()} if labels else {}

    feature_stats = endpoint.metadata.feature_stats or {}
    feature_names = endpoint.metadata.feature_names or []
    monitor_configuration = endpoint.metadata.monitor_configuration or {}

    client = get_v3io_client(endpoint=config.v3io_api)
    function = client.kv.update if update else client.kv.put
    function(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{endpoint.metadata.project}/{ENDPOINTS_TABLE_PATH}",
        key=endpoint.id,
        access_key=access_key,
        attributes={
            "endpoint_id": endpoint.id,
            "project": endpoint.metadata.project,
            "model": endpoint.spec.model,
            "function": endpoint.spec.function,
            "tag": endpoint.metadata.tag,
            "model_class": endpoint.spec.model_class or "",
            "labels": json.dumps(labels),
            "model_artifact": endpoint.metadata.model_artifact or "",
            "stream_path": endpoint.metadata.stream_path or "",
            "active": endpoint.status.active,
            "state": endpoint.status.state or "",
            "feature_stats": json.dumps(feature_stats),
            "feature_names": json.dumps(feature_names),
            "monitor_configuration": json.dumps(monitor_configuration),
            **searchable_labels,
        },
    )

    return endpoint


def deserialize_endpoint_from_kv(
    access_key: str, project: str, endpoint_id: str
) -> ModelEndpoint:

    logger.info(
        "Getting model endpoint record from kv", endpoint_id=endpoint_id,
    )

    client = get_v3io_client(endpoint=config.v3io_api)

    endpoint = client.kv.get(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
        key=endpoint_id,
        access_key=access_key,
        raise_for_status=RaiseForStatus.never,
    ).output.item

    if not endpoint:
        raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found")

    labels = endpoint.get("labels")
    feature_stats = endpoint.get("feature_stats")
    feature_names = endpoint.get("feature_names")
    monitor_configuration = endpoint.get("monitor_configuration")

    return ModelEndpoint.new(
        project=endpoint.get("project"),
        model=endpoint.get("model"),
        function=endpoint.get("function"),
        tag=endpoint.get("tag"),
        model_class=endpoint.get("model_class"),
        labels=_json_loads_if_not_none(labels),
        model_artifact=endpoint.get("model_artifact"),
        stream_path=endpoint.get("stream_path"),
        feature_stats=_json_loads_if_not_none(feature_stats),
        feature_names=_json_loads_if_not_none(feature_names),
        monitor_configuration=_json_loads_if_not_none(monitor_configuration),
        state=endpoint.get("state"),
        active=endpoint.get("active"),
    )


def deserialize_endpoint_state_from_kv(
    access_key: str, project: str, endpoint_id: str
) -> ModelEndpointState:

    logger.info(
        "Getting model endpoint record from kv", endpoint_id=endpoint_id,
    )

    client = get_v3io_client(endpoint=config.v3io_api)

    endpoint = client.kv.get(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
        key=endpoint_id,
        access_key=access_key,
        raise_for_status=RaiseForStatus.never,
    ).output.item

    labels = endpoint.get("labels")
    feature_stats = endpoint.get("feature_stats")
    feature_names = endpoint.get("feature_names")
    monitor_configuration = endpoint.get("monitor_configuration")

    drift_measures = endpoint.get("drift_measures")
    current_stats = endpoint.get("current_stats")

    return ModelEndpointState.new(
        endpoint=ModelEndpoint.new(
            project=endpoint.get("project"),
            model=endpoint.get("model"),
            function=endpoint.get("function"),
            tag=endpoint.get("tag"),
            model_class=endpoint.get("model_class"),
            labels=_json_loads_if_not_none(labels),
            model_artifact=endpoint.get("model_artifact"),
            stream_path=endpoint.get("stream_path"),
            feature_stats=_json_loads_if_not_none(feature_stats),
            feature_names=_json_loads_if_not_none(feature_names),
            monitor_configuration=_json_loads_if_not_none(monitor_configuration),
            state=endpoint.get("state"),
            active=endpoint.get("active"),
        ),
        first_request=endpoint.get("first_request"),
        last_request=endpoint.get("last_request"),
        accuracy=endpoint.get("accuracy"),
        error_count=endpoint.get("error_count"),
        drift_status=endpoint.get("drift_status"),
        drift_measures=_json_loads_if_not_none(drift_measures),
        current_stats=_json_loads_if_not_none(current_stats),
        # metrics -> Computed from TSDB
        # features -> Computed from `ModelEndpointState.endpoint.metadata.feature_stats` and `ModelEndpointState.current_stats`
    )


def _json_loads_if_not_none(field: Any):
    if field is None:
        return None
    return json.loads(field)


def _clean_feature_name(feature_name):
    return feature_name.replace(" ", "_").replace("(", "").replace(")", "")


def get_endpoint_metrics(
    access_key: str,
    project: str,
    endpoint_id: str,
    name: List[str],
    start: str = "now-1h",
    end: str = "now",
) -> Dict[str, Metric]:

    if not name:
        raise MLRunInvalidArgumentError("Metric names must be provided")

    metrics = list(map(string_to_tsdb_name, name))

    client = get_frames_client(
        token=access_key,
        address=config.v3io_framesd,
        container=config.model_endpoint_monitoring.container,
    )

    data = client.read(
        backend="tsdb",
        table=f"{project}/{ENDPOINT_EVENTS_TABLE_PATH}",
        columns=["endpoint_id", *metrics],
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    data_dict = data.to_dict()
    metrics_mapping = {}
    for metric in metrics:
        metric_data = data_dict.get(metric)
        if metric_data is None:
            continue

        values = [(str(timestamp), value) for timestamp, value in metric_data.items()]
        metrics_mapping[metric] = Metric(name=metric, values=values)
    return metrics_mapping


def string_to_tsdb_name(name: str) -> str:
    if name in {"latency_avg_1s", "average_latency", "latency"}:
        return "latency_avg_1s"
    elif name in {
        "predictions_per_second_count_1s",
        "predictions_per_second",
        "predictions",
    }:
        return "predictions_per_second_count_1s"
    else:
        raise MLRunInvalidArgumentError(f"Unsupported metric '{name}'")


def get_endpoint_features(
    feature_names: List[str],
    feature_stats: Optional[dict],
    current_stats: Optional[dict],
) -> List[Features]:
    safe_feature_stats = feature_stats or {}
    safe_current_stats = current_stats or {}

    features = []
    for name in feature_names:
        if feature_stats is not None and name not in feature_stats:
            logger.warn(f"Feature '{name}' missing from 'feature_stats'")
        if current_stats is not None and name not in current_stats:
            logger.warn(f"Feature '{name}' missing from 'current_stats'")
        f = Features.new(
            name, safe_feature_stats.get(name), safe_current_stats.get(name)
        )
        features.append(f)
    return features


def build_kv_cursor_filter_expression(
    project: str,
    function: Optional[str] = None,
    model: Optional[str] = None,
    tag: Optional[str] = None,
    labels: Optional[List[str]] = None,
):
    if not project:
        raise MLRunInvalidArgumentError("`project` can't be empty")

    filter_expression = [f"project=='{project}'"]

    if function:
        filter_expression.append(f"function=='{function}'")
    if model:
        filter_expression.append(f"model=='{model}'")
    if tag:
        filter_expression.append(f"tag=='{tag}'")
    if labels:
        for label in labels:

            if not label.startswith("_"):
                label = f"_{label}"

            if "==" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("==")))
                filter_expression.append(f"{lbl}=='{value}'")
            elif "=" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                filter_expression.append(f"{lbl}=='{value}'")
            else:
                filter_expression.append(f"exists({label})")

    return " AND ".join(filter_expression)


def get_access_key(request_headers: Mapping):
    access_key = request_headers.get("X-V3io-Session-Key")
    if not access_key:
        raise MLRunBadRequestError(
            "Request header missing 'X-V3io-Session-Key' parameter."
        )
    return access_key
