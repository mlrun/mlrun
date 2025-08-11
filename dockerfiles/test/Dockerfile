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
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG MLRUN_PYTHON_VERSION=3.9
ARG MLRUN_PIP_VERSION=25.0
ARG DOCKER_DEFAULT_PLATFORM=linux/amd64

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM --platform=${DOCKER_DEFAULT_PLATFORM} mlrun_common_image:${MLRUN_PYTHON_VERSION}

ARG MLRUN_PYTHON_VERSION
ARG MLRUN_PIP_VERSION

ENV NO_COLOR=1
LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

# Update OS packages:
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  -o Acquire::Check-Valid-Until=false  \
        g++ \
        graphviz \
        libpq-dev \
        libzmq3-dev \
        && \
        curl -fsSL https://get.docker.com -o get-docker.sh && \
        sh ./get-docker.sh && \
        rm -rf get-docker.sh /var/lib/apt/lists/*

# set initial git config
RUN git config --global user.email "test@mlrun.org" \
     && git config --global user.name "MLRun Test" \
     && git config --global init.defaultBranch "main"

RUN --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/pip \
    python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}


ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
# explicitly set the python version to ensure uv installs the correct version
# as per the python version or we will handle situtations where python 3.11 tries to install
# kfp 1.8 *even though it is explicitly set on locked file that kfp 1.8 is for python < 3.11*

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/test/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync  --require-hashes locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}

WORKDIR /mlrun
COPY *.txt *.md *.py *.toml ./
COPY mlrun mlrun
COPY dockerfiles/mlrun-api dockerfiles/mlrun-api

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install .[complete,dev-postgres] --python-version ${MLRUN_PYTHON_VERSION} && rm -rf /mlrun/build/

COPY pipeline-adapters pipeline-adapters
# Install the pipeline adapters development versions since we release them only after merging
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-common ./pipeline-adapters/mlrun-pipelines-kfp-v1-8 --python-version ${MLRUN_PYTHON_VERSION} && make -C pipeline-adapters clean

COPY Makefile .
COPY .dockerignore .
COPY docs docs
COPY server server
COPY tests tests
COPY automation automation
COPY dockerfiles dockerfiles
COPY examples examples

ENV PYTHONPATH="/mlrun/tests:/mlrun/automation:/mlrun/server/py/schemas/proto"
