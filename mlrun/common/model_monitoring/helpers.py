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

import sys
import typing
from datetime import datetime

import mlrun.common
import mlrun.common.schemas.model_monitoring.constants as mm_constants

FeatureStats = typing.NewType("FeatureStats", dict[str, dict[str, typing.Any]])
Histogram = typing.NewType("Histogram", list[list])
BinCounts = typing.NewType("BinCounts", list[int])
BinEdges = typing.NewType("BinEdges", list[float])

_MAX_FLOAT = sys.float_info.max
logger = mlrun.utils.create_logger(level="info", name="mm_helpers")


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = mlrun.platforms.iguazio.parse_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def get_kafka_topic(project: str, function_name: typing.Optional[str] = None) -> str:
    if (
        function_name is None
        or function_name == mm_constants.MonitoringFunctionNames.STREAM
    ):
        function_specifier = ""
    else:
        function_specifier = f"_{function_name}"

    return (
        f"monitoring_stream_{mlrun.mlconf.system_id}_{project}{function_specifier}_v1"
    )


def _get_counts(hist: Histogram) -> BinCounts:
    """Return the histogram counts"""
    return BinCounts(hist[0])


def _get_edges(hist: Histogram) -> BinEdges:
    """Return the histogram edges"""
    return BinEdges(hist[1])


def pad_hist(hist: Histogram) -> None:
    """
    Add [-inf, x_0] and [x_n, inf] bins to the histogram inplace unless present
    """
    counts = _get_counts(hist)
    edges = _get_edges(hist)

    is_padded = edges[0] == -_MAX_FLOAT and edges[-1] == _MAX_FLOAT
    if is_padded:
        return

    counts.insert(0, 0)
    edges.insert(0, -_MAX_FLOAT)

    counts.append(0)
    edges.append(_MAX_FLOAT)


def pad_features_hist(feature_stats: FeatureStats) -> None:
    """
    Given a feature statistics dictionary, pad the histograms with edges bins
    inplace to cover input statistics from -inf to inf.
    """
    hist_key = "hist"
    for feature in feature_stats.values():
        if hist_key in feature:
            pad_hist(Histogram(feature[hist_key]))


def get_model_endpoints_creation_task_status(
    server,
) -> tuple[
    mlrun.common.schemas.BackgroundTaskState,
    typing.Optional[datetime],
    typing.Optional[set[str]],
]:
    background_task = None
    background_task_state = mlrun.common.schemas.BackgroundTaskState.running
    background_task_check_timestamp = None
    model_endpoint_uids = None
    try:
        background_task = mlrun.get_run_db().get_project_background_task(
            server.project, server.model_endpoint_creation_task_name
        )
        background_task_check_timestamp = mlrun.utils.now_date()
        log_background_task_state(
            server, background_task.status.state, background_task_check_timestamp
        )
        background_task_state = background_task.status.state
    except mlrun.errors.MLRunNotFoundError:
        logger.warning(
            "Model endpoint creation task not found listing model endpoints",
            project=server.project,
            task_name=server.model_endpoint_creation_task_name,
        )
    if background_task is None:
        model_endpoints = mlrun.get_run_db().list_model_endpoints(
            project=server.project,
            function_name=server.function_name,
            function_tag=server.function_tag,
            tsdb_metrics=False,
        )
        if model_endpoints:
            model_endpoint_uids = {
                endpoint.metadata.uid for endpoint in model_endpoints.endpoints
            }
            logger.info(
                "Model endpoints found after background task not found, model monitoring will monitor "
                "events",
                project=server.project,
                function_name=server.function_name,
                function_tag=server.function_tag,
                uids=model_endpoint_uids,
            )
            background_task_state = mlrun.common.schemas.BackgroundTaskState.succeeded
        else:
            logger.warning(
                "Model endpoints not found after background task not found, model monitoring will not "
                "monitor events",
                project=server.project,
                function_name=server.function_name,
                function_tag=server.function_tag,
            )
            background_task_state = mlrun.common.schemas.BackgroundTaskState.failed
    return background_task_state, background_task_check_timestamp, model_endpoint_uids


def log_background_task_state(
    server,
    background_task_state: mlrun.common.schemas.BackgroundTaskState,
    background_task_check_timestamp: typing.Optional[datetime],
):
    logger.info(
        "Checking model endpoint creation task status",
        task_name=server.model_endpoint_creation_task_name,
    )
    if (
        background_task_state
        in mlrun.common.schemas.BackgroundTaskState.terminal_states()
    ):
        logger.info(
            f"Model endpoint creation task completed with state {background_task_state}"
        )
    else:  # in progress
        logger.info(
            f"Model endpoint creation task is still in progress with the current state: "
            f"{background_task_state}. Events will not be monitored for the next "
            f"{mlrun.mlconf.model_endpoint_monitoring.model_endpoint_creation_check_period} seconds",
            function_name=server.function_name,
            background_task_check_timestamp=background_task_check_timestamp.isoformat(),
        )
