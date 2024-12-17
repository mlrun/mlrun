# Copyright 2024 Iguazio
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
ARG MLRUN_DOCKER_REGISTRY
ARG MLRUN_VERSION=unstable
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest
# Compute the base image name, adding a slash only if the registry is not empty
ARG BASE_IMAGE=${MLRUN_DOCKER_REGISTRY:+${MLRUN_DOCKER_REGISTRY}/}mlrun/ml-base

FROM ${MLRUN_UV_IMAGE} AS uv-image

FROM ${BASE_IMAGE}:${MLRUN_VERSION}
ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun-kfp/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync --require-hashes locked-requirements.txt

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
uv pip install .[kfp18]
