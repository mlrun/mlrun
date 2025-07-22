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
ARG MLRUN_PYTHON_VERSION=3.9
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim

ENV NO_COLOR=1

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        curl \
        apt-transport-https \
        ca-certificates \
        g++ \
        git \
        git-core \
        gnupg2 \
        graphviz \
        make \
        software-properties-common \
        libpq-dev \
        python3-dev && \
    curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh ./get-docker.sh && \
    rm -rf get-docker.sh


# set initial git config
RUN git config --global user.email "test@mlrun.org" \
     && git config --global user.name "MLRun Test" \
     && git config --global init.defaultBranch "main"

ARG MLRUN_PIP_VERSION=25.0
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}


ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
ARG MLRUN_PYTHON_VERSION=3.9
# explicitly set the python version to ensure uv installs the correct version
# as per the python version or we will handle situtations where python 3.11 tries to install
# kfp 1.8 *even though it is explicitly set on locked file that kfp 1.8 is for python < 3.11*

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/test/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync  --require-hashes locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}
WORKDIR /mlrun
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
    uv pip install .[complete,dev-postgres] --python-version ${MLRUN_PYTHON_VERSION}

COPY pipeline-adapters pipeline-adapters
# Install the pipeline adapters development versions since we release them only after merging
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-common ./pipeline-adapters/mlrun-pipelines-kfp-v1-8 --python-version ${MLRUN_PYTHON_VERSION}

RUN find /mlrun \
     \( -type f -name '*.pyc' \
        -o -type d -name '__pycache__' \
        -o -path '*/tests' \
        -o -path '*/build' \
        -o -path '*/dist' \) \
     -exec rm -rf {} + \
 && rm -rf /tmp/*

COPY Makefile .
COPY docs docs
COPY server server
COPY tests tests
COPY automation automation
COPY dockerfiles dockerfiles
COPY examples examples

ENV PYTHONPATH="/mlrun/tests:/mlrun/automation:/mlrun/server/api/proto"
