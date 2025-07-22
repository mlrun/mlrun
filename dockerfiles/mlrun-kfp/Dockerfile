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

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim

# Update OS packages:
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gcc \
        wget \
        git-core \
        unzip


ARG MLRUN_PIP_VERSION=25.0
RUN --mount=type=cache,target=/root/.cache/pip \
  python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}
ARG MLRUN_PYTHON_VERSION=3.9

WORKDIR /mlrun

ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun-kfp/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync --require-hashes locked-requirements.txt

WORKDIR /tmp/mlrun
COPY *requirements.txt .
COPY setup.py .
COPY packages.py .
COPY README.md .
COPY dependencies.py .
COPY pyproject.toml .
COPY mlrun mlrun
COPY dockerfiles/mlrun-api/requirements.txt dockerfiles/mlrun-api/requirements.txt
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install .  && rm -rf /tmp/mlrun

COPY pipeline-adapters pipeline-adapters
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-common ./pipeline-adapters/mlrun-pipelines-kfp-v1-8[kfp]  && rm -rf /tmp/mlrun

# Remove server related files that are not needed
# Remove ensurepip folders from both conda and system python
RUN rm -rf /usr/local/lib/python${MLRUN_PYTHON_VERSION}/ensurepip /tmp/mlrun
