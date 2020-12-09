from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query
from pandas import Grouper
from v3io.dataplane import RaiseForStatus

from mlrun.api import schemas
from mlrun.config import config
from mlrun.errors import MLRunConflictError, MLRunNotFoundError
from mlrun.utils import logger
from mlrun.utils.helpers import get_model_endpoint_id
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client

ENDPOINTS_KV_TABLE = "endpoints_kv"
ENDPOINTS_TSDB_TABLE = "endpoints_tsdb"

router = APIRouter()


@router.post(
    "/projects/{project}/model-endpoints", response_model=schemas.ModelEndpoint
)
def create_endpoint(project: str, endpoint_identifies: schemas.ModelEndpoint):
    """
    This function is used to write active model endpoints to endpoints table.
    """

    endpoint_id = get_model_endpoint_id(
        project,
        endpoint_identifies.model,
        endpoint_identifies.function,
        endpoint_identifies.tag,
    )

    if _get_endpoint_by_id(endpoint_id):
        url = f"/projects/{endpoint_identifies.project}/model-endpoints/{endpoint_id}"
        raise MLRunConflictError(f"Adding an already-existing ModelEndpoint - {url}")

    logger.info(f"Creating endpoint {endpoint_id} table...")

    labels = _encode_labels(endpoint_identifies.labels)

    attributes = {
        "project": project,
        "function": endpoint_identifies.function,
        "model": endpoint_identifies.model,
        "tag": endpoint_identifies.tag,
        "model_class": endpoint_identifies.model_class,
        **labels,
    }

    get_v3io_client().kv.put(
        container=config.model_endpoint_monitoring_container,
        table_path=ENDPOINTS_KV_TABLE,
        key=endpoint_id,
        attributes=attributes,
    )

    logger.info(f"Model endpoint {endpoint_id} table created.")

    return schemas.ModelEndpoint(**attributes)


@router.delete("/projects/{project}/model-endpoints/{endpoint_id}")
def delete_endpoint(project: str, endpoint_id: str):
    logger.info(f"Deleting model endpoint {endpoint_id} table...")
    get_v3io_client().kv.delete(
        container=config.model_endpoint_monitoring_container,
        table_path=ENDPOINTS_KV_TABLE,
        key=endpoint_id,
    )
    logger.info(f"Model endpoint {endpoint_id} table deleted.")


@router.get(
    "/projects/{project}/model-endpoints", response_model=schemas.ModelEndpointsList
)
def list_endpoints(
    project: str,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    labels: List[str] = Query([], alias="label"),
):
    filter_expression = _build_filter_expression(project, function, model, tag, labels)
    # TODO: call async version of v3io_client
    records = (
        get_v3io_client()
        .kv.new_cursor(
            container=config.model_endpoint_monitoring_container,
            table_path=ENDPOINTS_KV_TABLE,
            filter_expression=filter_expression,
        )
        .all()
    )

    endpoints = []
    for record in records:
        labels = {
            k: record[k] for k in record.keys() if k.startswith("_") and k != "__name"
        }
        model_endpoint = schemas.ModelEndpoint(
            project=record.get("project"),
            model=record.get("model"),
            function=record.get("function"),
            tag=record.get("tag"),
            model_class=record.get("model_class"),
            labels=_decode_labels(labels),
        )
        endpoints.append(model_endpoint)

    return schemas.ModelEndpointsList(endpoints=endpoints)


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    response_model=schemas.ModelEndpointState,
)
def get_endpoint_state(
    project: str,
    endpoint_id: str,
    start: str = Query("now-5m"),
    end: str = Query("now"),
):
    endpoint = _get_endpoint_by_id(endpoint_id)

    if not endpoint:
        raise MLRunNotFoundError(f"Model endpoint {endpoint_id} not found")

    # If model state was collected successfully, try to collect predictions made in time frame
    time_series_data = get_frames_client().read(
        backend="tsdb",
        table=ENDPOINTS_TSDB_TABLE,
        columns=["endpoint_id", "microsec", "requests", "predictions"],
        start=start,
        end=end,
    )

    average_latency = None  # TODO: Compute
    requests_histogram = None  # TODO: Compute
    predictions_histogram = None  # TODO: Compute
    feature_details = None

    if not time_series_data.empty:
        predictions_per_second_data = (
            time_series_data["predictions"].groupby(Grouper(freq="1s")).count()
        )
        requests_per_second_data = (
            time_series_data["requests"].groupby(Grouper(freq="1s")).count()
        )
        average_latency_data = (
            time_series_data["microsec"].groupby(Grouper(freq="1s")).mean().dropna()
        )

        predictions_per_second_data.index = predictions_per_second_data.index.format()
        requests_per_second_data.index = requests_per_second_data.index.format()
        average_latency = average_latency_data.mean()

        # "predictions_per_second": predictions_per_second_data.to_dict(),
        # "average_latency": average_latency.mean(),

    labels = {
        k: endpoint[k] for k in endpoint.keys() if k.startswith("_") and k != "__name"
    }

    return schemas.ModelEndpointState(
        model_endpoint=schemas.ModelEndpoint(
            project=endpoint.get("project"),
            model=endpoint.get("model"),
            function=endpoint.get("function"),
            tag=endpoint.get("tag"),
            model_class=endpoint.get("model_class"),
            labels=_decode_labels(labels),
        ),
        first_request=endpoint.get("first_request"),
        last_request=endpoint.get("last_request"),
        accuracy=endpoint.get("accuracy"),
        error_count=endpoint.get("error_count"),
        alert_count=endpoint.get("alert_count"),
        drift_status=endpoint.get("drift_status"),
        average_latency=average_latency,
        requests_histogram=requests_histogram,
        predictions_histogram=predictions_histogram,
        feature_details=feature_details,
    )


def _encode_labels(labels: Optional[List[str]]) -> Dict[str, Any]:
    if not labels:
        return {}

    processed_labels = {}
    for label in labels:
        lbl, val = list(map(lambda x: x.strip(), label.split("==")))
        processed_labels[f"_{lbl}"] = val

    return processed_labels


def _decode_labels(labels: Dict[str, Any]) -> List[str]:
    if not labels:
        return []
    return [f"{lbl.lstrip('_')}=={val}" for lbl, val in labels.items()]


def _build_filter_expression(
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

            if "==" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("==")))
                filter_expression.append(f"{lbl}=='{value}'")

    return " AND ".join(filter_expression)


def _get_endpoint_by_id(endpoint_id: str):
    endpoint = (
        get_v3io_client()
        .kv.get(
            container=config.model_endpoint_monitoring_container,
            table_path=ENDPOINTS_KV_TABLE,
            key=endpoint_id,
            raise_for_status=RaiseForStatus.never,
        )
        .output.item
    )
    return endpoint
