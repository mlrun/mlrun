from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, Query
from pandas import Grouper
from v3io.dataplane import Client as V3IOClient
from v3io.dataplane import RaiseForStatus
from v3io_frames import Client as FramesClient

from mlrun.api import schemas
from mlrun.utils import logger

DEFAULT_CONTAINER = "monitoring"
ENDPOINTS_TABLE = "endpoints"


@dataclass()
class EndpointKey:
    project: str
    function: str
    model: str
    tag: str
    model_class: Optional[str] = None,
    hash: Optional[str] = None

    def __post_init__(self):
        self.hash: str = f"{self.project}_{self.function}_{self.model}_{self.tag}"

    def __str__(self):
        return self.hash


# TODO: Can be done nicer, also this code assumes environment parameters exist for initializing both frames and v3io
_v3io_client: Optional[V3IOClient, None] = None
_frames_client: Optional[FramesClient, None] = None

router = APIRouter()


@router.post("/projects/{project}/models")
def create_endpoint(project: str, endpoint_identifies: schemas.EndpointIdentifiers):
    """
    this function should be called on creating a model server via v2_model_server
    """

    key = EndpointKey(
        project,
        endpoint_identifies.function,
        endpoint_identifies.model,
        endpoint_identifies.tag,
    )

    if get_endpoint(key.hash) is not None:
        logger.info(f"Endpoint {key} already exists")
        return

    logger.info(f"Creating endpoint [{key}]...")

    get_v3io_client().kv.put(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINTS_TABLE,
        key=key.hash,
        attributes={
            "project": project,
            "function": endpoint_identifies.function,
            "model": endpoint_identifies.model,
            "version": endpoint_identifies.tag,
        },
    )

    logger.info(f"Endpoint [{key}] created.")


@router.delete("projects/{project}/models/{function}:{model}/references/{tag}")
def delete_endpoint(project: str, function: str, model: str, version: str):
    key = EndpointKey(project, function, model, version)
    logger.info(f"Deleting endpoint [{key}]...")
    get_v3io_client().kv.delete(container=DEFAULT_CONTAINER, table_path=ENDPOINTS_TABLE, key=key.hash)
    logger.info(f"Endpoint [{key}] deleted.")


@router.put("projects/{project}/models/{function}:{model}/references/{tag}")
def update_endpoint(
    project: str,
    function: str,
    model: str,
    tag: str,
    updated_identifiers: schemas.EndpointIdentifiers
):
    key = EndpointKey(project, function, model, tag)

    logger.info(f"Updating endpoint [{key}...]")

    get_v3io_client().kv.delete(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINTS_TABLE,
        key=key.hash
    )

    new_key = EndpointKey(
        updated_identifiers.project or project,
        updated_identifiers.function or function,
        updated_identifiers.model or model,
        updated_identifiers.tag or tag
    )

    get_v3io_client().kv.put(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINTS_TABLE,
        key=new_key.hash,
        attributes={
            "project": updated_identifiers.project or project,
            "function": updated_identifiers.function or function,
            "model": updated_identifiers.model or model,
            "tag": updated_identifiers.tag or tag
        }
    )

    logger.info(f"Endpoint [{key}] update to [{new_key}]")


@router.post("/projects/{project}/models", response_model=schemas.EndpointIdentifiers)
def list_endpoints(
        project: str,
        function: str = Query(None),
        model: str = Query(None),
        tag: str = Query(None)
):

    filter_expression = [f"project=='{project}'"]

    if function:
        filter_expression.append(f"function=='{function}'")
    if model:
        filter_expression.append(f"model=='{model}'")
    if tag:
        filter_expression.append(f"tag=='{tag}'")

    filter_expression = " AND ".join(filter_expression)

    return (
        get_v3io_client()
        .kv.new_cursor(
            container="monitoring",
            table_path="endpoints",
            filter_expression=filter_expression,
        )
        .all()
    )


def get_endpoint(
    endpoint_key: str,
    with_state: bool = False,
    with_ts_values: bool = False,
    start_time: str = "now-5m",
    end_time: str = "now",
    verbose: bool = False,
):
    client = get_v3io_client()

    endpoint = client.kv.get(
        container="monitoring",
        table_path="endpoints",
        key=endpoint_key,
        raise_for_status=RaiseForStatus.never,
    ).output.item

    if not endpoint:
        if verbose:
            logger.info(f"Endpoint {endpoint_key} not found")
        return None
    else:
        if verbose:
            logger.info(f"Got endpoint [{endpoint}]")

        # Collect model state
        if with_state:
            state = client.kv.get(
                container="monitoring",
                table_path="model_state",
                key=endpoint_key,
                raise_for_status=RaiseForStatus.never,
            ).output.item

            if state:
                endpoint.update(state)

                # If model state was collected successfully, try to collect predictions made in time frame
                time_series_data = get_frames_client().read(
                    backend="tsdb",
                    table="model_event_log",
                    columns=["model_hash", "microsec"],
                    start=start_time,
                    end=end_time,
                )

                if not time_series_data.empty:
                    predictions_per_second = (
                        time_series_data["model_hash"]
                        .groupby(Grouper(freq="1s"))
                        .count()
                    )

                    predictions_per_second.index = predictions_per_second.index.format()

                    average_latency = (
                        time_series_data["microsec"]
                        .groupby(Grouper(freq="1s"))
                        .mean()
                        .dropna()
                    )

                    average_latency.index = average_latency.index.format()

                    if with_ts_values:
                        endpoint = {
                            **endpoint,
                            "predictions_per_second": predictions_per_second.to_dict(),
                            "average_latency": average_latency.to_dict(),
                        }
                    else:
                        endpoint = {
                            **endpoint,
                            "predictions_per_second": predictions_per_second.mean(),
                            "average_latency": average_latency.mean(),
                        }

        return endpoint


def endpoint_summary(start_time: str, end_time: str = "now", verbose=False):
    summary = defaultdict(float)
    for endpoint in list_endpoints():

        key = EndpointKey(
            endpoint["project"],
            endpoint["function"],
            endpoint["model"],
            endpoint["version"],
        )

        details = get_endpoint(
            key.hash,
            with_state=True,
            with_ts_values=False,
            verbose=verbose,
            start_time=start_time,
            end_time=end_time,
        )

        summary["predictions_per_second"] += details["predictions_per_second"]
        summary["average_latency"] += details["average_latency"]
        summary["drift_alerts"] += endpoint.get("drift_alerts", 0)
        summary["requests_per_second"] += endpoint.get("requests_per_second", 0)
        summary["endpoints"] += 1
    return summary


def get_frames_client() -> FramesClient:
    global _frames_client
    if _frames_client is None:
        _frames_client = FramesClient(container=DEFAULT_CONTAINER, should_check_version=False)
    return _frames_client


def get_v3io_client() -> V3IOClient:
    global _v3io_client
    if _v3io_client is None:
        _v3io_client = V3IOClient()
    return _v3io_client
