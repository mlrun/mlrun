from collections import defaultdict

from fastapi import APIRouter, Query
from pandas import Grouper
from v3io.dataplane import RaiseForStatus

from mlrun.api import schemas
from mlrun.monitoring.clients import get_v3io_client, get_frames_client
from mlrun.monitoring.utils import build_endpoint_key
from mlrun.monitoring.constants import DEFAULT_CONTAINER, ENDPOINTS_TABLE
from mlrun.utils import logger

router = APIRouter()


@router.post("/projects/{project}/models")
def create_endpoint(project: str, endpoint_identifies: schemas.EndpointIdentifiers):
    """
    this function should be called on creating a model server via v2_model_server
    """

    key = build_endpoint_key(
        project,
        endpoint_identifies.function,
        endpoint_identifies.model,
        endpoint_identifies.tag,
    )

    if get_endpoint(key) is not None:
        logger.info(f"Endpoint {key} already exists")
        return

    logger.info(f"Creating endpoint [{key}]...")

    get_v3io_client().kv.put(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINTS_TABLE,
        key=key,
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
    key = build_endpoint_key(project, function, model, version)
    logger.info(f"Deleting endpoint [{key}]...")
    get_v3io_client().kv.delete(container=DEFAULT_CONTAINER, table_path=ENDPOINTS_TABLE, key=key)
    logger.info(f"Endpoint [{key}] deleted.")


@router.put("projects/{project}/models/{function}:{model}/references/{tag}")
def update_endpoint(
    project: str,
    function: str,
    model: str,
    tag: str,
    updated_identifiers: schemas.EndpointIdentifiers
):
    key = build_endpoint_key(project, function, model, tag)

    logger.info(f"Updating endpoint [{key}...]")

    get_v3io_client().kv.delete(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINTS_TABLE,
        key=key
    )

    new_key = build_endpoint_key(
        updated_identifiers.project or project,
        updated_identifiers.function or function,
        updated_identifiers.model or model,
        updated_identifiers.tag or tag
    )

    get_v3io_client().kv.put(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINTS_TABLE,
        key=key,
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

        key = build_endpoint_key(
            endpoint["project"],
            endpoint["function"],
            endpoint["model"],
            endpoint["version"],
        )

        details = get_endpoint(
            key,
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
