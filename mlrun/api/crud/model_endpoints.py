import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import Request
from v3io.dataplane import RaiseForStatus

from mlrun.api.schemas import (
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpoint,
    ModelEndpointState,
    ModelEndpointUpdatePayload,
    Features,
    Metric,
    ObjectStatus,
)
from mlrun.config import config
from mlrun.errors import (
    MLRunConflictError,
    MLRunNotFoundError,
    MLRunInvalidArgumentError,
    MLRunBadRequestError,
)
from mlrun.utils.helpers import logger
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client
from mlrun.artifacts import get_model

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

    @classmethod
    def from_string(cls, name: str) -> "TimeMetric":
        if name in {"microsec", "latency"}:
            return cls(
                tsdb_column="latency_avg_1s",
                metric_name="average_latency",
                headers=["timestamp", "average"],
            )
        elif name in {"preds", "predictions"}:
            return cls(
                tsdb_column="predictions_per_second_count_1s",
                metric_name="predictions_per_second",
                headers=["timestamp", "count"],
            )
        else:
            raise NotImplementedError(f"Unsupported metric '{name}'")


class ModelEndpoints:
    @staticmethod
    def register_endpoint(
        access_key: str,
        project: str,
        model: str,
        function: str,
        model_class: str,
        model_artifact: str,
        stream_path: str,
        tag: str = "latest",
        labels: Optional[dict] = None,
        active: bool = True,
    ):
        """
        Writes endpoint data to KV, a prerequisite for initializing the monitoring process

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param model: The name of the model
        :param function: The function that
        :param model_class:
        :param model_artifact:
        :param stream_path:
        :param tag:
        :param labels:
        :param active:
        """
        model_endpoint = ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=project,
                tag=tag,
                labels=labels or {},
                model_artifact=model_artifact,
                stream_path=stream_path,
            ),
            spec=ModelEndpointSpec(
                model=model, function=function, model_class=model_class,
            ),
            status=ObjectStatus(state="registered"),
            active=active,
        )
        logger.info("Getting model feature data...")

        model_obj = get_model(model_artifact)
        feature_stats = model_obj[1].feature_stats
        feature_stats = {_clean_feature_name(k): v for k, v in feature_stats.items()}

        logger.info("Registering model endpoint...", endpoint_id=model_endpoint.id)

        client = get_v3io_client(endpoint=config.v3io_api)

        client.kv.put(
            container=config.model_endpoint_monitoring.container,
            table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
            key=model_endpoint.id,
            access_key=access_key,
            attributes={
                "project": model_endpoint.metadata.project,
                "tag": model_endpoint.metadata.tag,
                "model_artifact": model_endpoint.metadata.model_artifact,
                "stream_path": model_endpoint.metadata.stream_path,
                "model": model_endpoint.spec.model,
                "function": model_endpoint.spec.function,
                "model_class": model_endpoint.spec.model_class,
                "status": model_endpoint.status.state,
                "active": model_endpoint.active,
                "endpoint_id": model_endpoint.id,
                "base_stats": json.dumps(feature_stats)
            },
        )

        logger.info("Model endpoint registered", endpoint_id=model_endpoint.id)

        return model_endpoint

    @staticmethod
    def clear_endpoint_record(access_key: str, project: str, endpoint_id: str):
        """
        Clears the KV data of a given model endpoint

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        """
        verify_endpoint(project, endpoint_id)

        logger.info("Clearing model endpoint table", endpoint_id=endpoint_id)
        client = get_v3io_client(endpoint=config.v3io_api)
        client.kv.delete(
            container=config.model_endpoint_monitoring.container,
            table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
            key=endpoint_id,
            access_key=access_key,
        )

        logger.info("Model endpoint table deleted", endpoint_id=endpoint_id)

    @staticmethod
    def update_endpoint_record(
        access_key: str,
        project: str,
        endpoint_id: str,
        payload: ModelEndpointUpdatePayload,
    ):
        """
        Updates the KV data of a given model endpoint

        :param access_key: V3IO access key for managing user permissions
        :param project: The name of the project
        :param endpoint_id: The id of the endpoint
        :param payload: The parameters that are available for update
        """
        verify_endpoint(project, endpoint_id)

        payload_dict = payload.as_dict()

        if not payload_dict:
            raise MLRunInvalidArgumentError(
                "Update payload must contain at least one field to update"
            )

        logger.info("Updating model endpoint table", endpoint_id=endpoint_id)
        client = get_v3io_client(endpoint=config.v3io_api)

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
            attributes=payload_dict,
        )
        logger.info("Model endpoint table updated", endpoint_id=endpoint_id)

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

        client = get_v3io_client(endpoint=config.v3io_api)
        cursor = client.kv.new_cursor(
            container=config.model_endpoint_monitoring.container,
            table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
            access_key=access_key,
            attribute_names=ENDPOINT_TABLE_ATTRIBUTES,
            filter_expression=_build_kv_cursor_filter_expression(
                project, function, model, tag, labels
            ),
        )
        endpoints = cursor.all()

        endpoint_state_list = []
        for endpoint in endpoints:
            endpoint_metrics = {}
            if metrics:
                endpoint_metrics = _get_endpoint_metrics(
                    access_key=access_key,
                    project=project,
                    endpoint_id=endpoint.get("id"),
                    name=metrics,
                    start=start,
                    end=end,
                )

            # Collect labels (by convention labels are labeled with underscore '_'), ignore builtin '__name' field
            state = ModelEndpointState(
                endpoint=ModelEndpoint(
                    metadata=ModelEndpointMetadata(
                        project=endpoint.get("project"),
                        tag=endpoint.get("tag"),
                        labels=json.loads(endpoint.get("labels")),
                    ),
                    spec=ModelEndpointSpec(
                        model=endpoint.get("model"),
                        function=endpoint.get("function"),
                        model_class=endpoint.get("model_class"),
                    ),
                    status=ObjectStatus(state="active"),
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
        """
        verify_endpoint(project, endpoint_id)

        endpoint = get_endpoint_kv_record_by_id(
            access_key, project, endpoint_id, ENDPOINT_TABLE_ATTRIBUTES,
        )

        if not endpoint:
            url = f"/projects/{project}/model-endpoints/{endpoint_id}"
            raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found - {url}")

        endpoint_metrics = None
        if metrics:
            endpoint_metrics = _get_endpoint_metrics(
                access_key=access_key,
                project=project,
                endpoint_id=endpoint_id,
                start=start,
                end=end,
                name=metrics,
            )

        return ModelEndpointState(
            endpoint=ModelEndpoint(
                metadata=ModelEndpointMetadata(
                    project=endpoint.get("project"),
                    tag=endpoint.get("tag"),
                    labels=json.loads(endpoint.get("labels", "")),
                ),
                spec=ModelEndpointSpec(
                    model=endpoint.get("model"),
                    function=endpoint.get("function"),
                    model_class=endpoint.get("model_class"),
                ),
                status=ObjectStatus(state="active"),
            ),
            first_request=endpoint.get("first_request"),
            last_request=endpoint.get("last_request"),
            error_count=endpoint.get("error_count"),
            drift_status=endpoint.get("drift_status"),
            metrics=endpoint_metrics,
        )


def _clean_feature_name(feature_name):
    return feature_name.replace(" ", "_").replace("(", "").replace(")", "")


def _get_endpoint_metrics(
    access_key: str,
    project: str,
    endpoint_id: str,
    name: List[str],
    start: str = "now-1h",
    end: str = "now",
) -> Dict[str, Metric]:

    if not name:
        raise MLRunInvalidArgumentError("Metric names must be provided")

    try:
        metrics = [TimeMetric.from_string(n) for n in name]
    except NotImplementedError as e:
        raise MLRunInvalidArgumentError(str(e))

    # Columns must have at least an endpoint_id attribute for frames' filter expression
    columns = ["endpoint_id"]

    for metric in metrics:
        columns.append(metric.tsdb_column)

    client = get_frames_client(
        token=access_key,
        address=config.v3io_framesd,
        container=config.model_endpoint_monitoring.container,
    )

    data = client.read(
        backend="tsdb",
        table=f"{project}/{ENDPOINT_EVENTS_TABLE_PATH}",
        columns=columns,
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    metrics = [time_metric.transform_df_to_metric(data) for time_metric in metrics]
    metrics = [metric for metric in metrics if metric is not None]
    metrics = {metric.name: metric for metric in metrics}
    return metrics


def _get_endpoint_features(
    project: str, endpoint_id: str, features: Optional[str]
) -> List[Features]:
    if not features:
        url = f"projects/{project}/model-endpoints/{endpoint_id}/features"
        raise MLRunNotFoundError(f"Endpoint features not found' - {url}")

    parsed_features: List[dict] = json.loads(features) if features is not None else []
    feature_objects = [Features(**feature) for feature in parsed_features]
    return feature_objects


def decode_labels(labels: Dict[str, Any]) -> List[str]:
    if not labels:
        return []
    return [f"{lbl.lstrip('_')}=={val}" for lbl, val in labels.items()]


def _build_kv_cursor_filter_expression(
    project: str,
    function: Optional[str],
    model: Optional[str],
    tag: Optional[str],
    labels: Optional[List[str]],
):
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

            if "=" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                filter_expression.append(f"{lbl}=='{value}'")
            else:
                filter_expression.append(f"exists({label})")

    return " AND ".join(filter_expression)


def get_endpoint_kv_record_by_id(
    access_key: str,
    project: str,
    endpoint_id: str,
    attribute_names: Optional[List[str]] = None,
) -> Dict[str, Any]:

    client = get_v3io_client(endpoint=config.v3io_api)

    endpoint = client.kv.get(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
        key=endpoint_id,
        access_key=access_key,
        attribute_names=attribute_names or "*",
        raise_for_status=RaiseForStatus.never,
    ).output.item
    return endpoint


def verify_endpoint(project, endpoint_id):
    endpoint_id_project, _ = endpoint_id.split(".")
    if endpoint_id_project != project:
        raise MLRunConflictError(
            f"project: {project} and endpoint_id: {endpoint_id} missmatch."
        )


def get_access_key(request: Request):
    access_key = request.headers.get("X-V3io-Session-Key")
    if not access_key:
        raise MLRunBadRequestError(
            "Request header missing 'X-V3io-Session-Key' parameter."
        )
    return access_key
