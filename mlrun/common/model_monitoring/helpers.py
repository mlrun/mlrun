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

import mlrun.common
import mlrun.common.schemas.model_monitoring.constants as mm_constants
from mlrun.common.schemas.model_monitoring import (
    EndpointUID,
    FunctionURI,
    VersionedModel,
)

FeatureStats = typing.NewType("FeatureStats", dict[str, dict[str, typing.Any]])
Histogram = typing.NewType("Histogram", list[list])
BinCounts = typing.NewType("BinCounts", list[int])
BinEdges = typing.NewType("BinEdges", list[float])

_MAX_FLOAT = sys.float_info.max


def create_model_endpoint_uid(function_uri: str, versioned_model: str):
    function_uri = FunctionURI.from_string(function_uri)
    versioned_model = VersionedModel.from_string(versioned_model)

    if (
        not function_uri.project
        or not function_uri.function
        or not versioned_model.model
    ):
        raise ValueError("Both function_uri and versioned_model have to be initialized")

    uid = EndpointUID(
        function_uri.project,
        function_uri.function,
        function_uri.tag,
        function_uri.hash_key,
        versioned_model.model,
        versioned_model.version,
    )

    return uid


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = mlrun.platforms.iguazio.parse_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def parse_monitoring_stream_path(
    stream_uri: str, project: str, function_name: typing.Optional[str] = None
):
    if stream_uri.startswith("kafka://"):
        if "?topic" in stream_uri:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Custom kafka topic is not allowed"
            )
        # Add topic to stream kafka uri
        if (
            function_name is None
            or function_name == mm_constants.MonitoringFunctionNames.STREAM
        ):
            stream_uri += f"?topic=monitoring_stream_{project}"
        else:
            stream_uri += f"?topic=monitoring_stream_{project}_{function_name}"

    elif stream_uri.startswith("v3io://") and mlrun.mlconf.is_ce_mode():
        # V3IO is not supported in CE mode, generating a default http stream path
        if function_name is None:
            stream_uri = (
                mlrun.mlconf.model_endpoint_monitoring.default_http_sink.format(
                    project=project, namespace=mlrun.mlconf.namespace
                )
            )
        else:
            stream_uri = (
                mlrun.mlconf.model_endpoint_monitoring.default_http_sink_app.format(
                    project=project,
                    application_name=function_name,
                    namespace=mlrun.mlconf.namespace,
                )
            )
    return stream_uri


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
