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

# Install docker
RUN apt-get update -qq >/dev/null \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://get.docker.com -o get-docker.sh \
    && sh ./get-docker.sh

# Install other dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        g++ \
        git \
        git-core \
        gnupg2 \
        graphviz \
        make \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# set initial git config
RUN git config --global user.email "test@mlrun.org" \
     && git config --global user.name "MLRun Test" \
     && git config --global init.defaultBranch "main"

ARG MLRUN_PIP_VERSION=24.2
RUN python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}

WORKDIR /mlrun

ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy
ARG MLRUN_PYTHON_VERSION=3.9
# explicitly set the python version to ensure uv installs the correct version
# as per the python version or we will handle situtations where python 3.11 tries to install
# kfp 1.8 *even though it is explicitly set on locked file that kfp 1.8 is for python < 3.11*
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/test/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync --require-hashes locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}

COPY . .

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[complete] --python-version ${MLRUN_PYTHON_VERSION}
