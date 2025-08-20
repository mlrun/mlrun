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

ARG MLRUN_PYTHON_VERSION=3.9
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG MLRUN_PIP_VERSION=25.0
ARG DOCKER_DEFAULT_PLATFORM=linux/amd64

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM --platform=${DOCKER_DEFAULT_PLATFORM} mlrun_common_image:${MLRUN_PYTHON_VERSION}

ARG MLRUN_PYTHON_VERSION
ARG MLRUN_PIP_VERSION

RUN --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/pip \
  python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}

WORKDIR /mlrun

ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun-kfp/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync --require-hashes locked-requirements.txt

WORKDIR /tmp/mlrun
COPY *.txt *.md *.py *.toml ./
COPY mlrun mlrun
COPY dockerfiles/mlrun-api dockerfiles/mlrun-api

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install .[kfp18] --python-version ${MLRUN_PYTHON_VERSION}

COPY pipeline-adapters pipeline-adapters
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-common ./pipeline-adapters/mlrun-pipelines-kfp-v1-8[kfp]

# Remove server related files that are not needed
# Remove ensurepip folders from both conda and system python
RUN rm -rf /usr/local/lib/python${MLRUN_PYTHON_VERSION}/ensurepip /tmp/mlrun
