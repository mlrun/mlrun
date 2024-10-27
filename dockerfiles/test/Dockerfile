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

FROM gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim

ENV PIP_NO_CACHE_DIR=1

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

COPY ./requirements.txt ./
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./dockerfiles/mlrun-api/requirements.txt ./mlrun-api-requirements.txt
COPY ./dev-requirements.txt ./
COPY ./docs/requirements.txt ./docs-requirements.txt

RUN pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-api-requirements.txt

# if we give all the requirements together pip resolver go nuts so separating between the package and the image
# requirements
COPY ./dev-requirements.txt ./
RUN python -m pip install \
    -r dev-requirements.txt

COPY ./docs/requirements.txt ./docs-requirements.txt
RUN python -m pip install \
    -r docs-requirements.txt

COPY . .

RUN pip install -e .[complete] &&\
    pip install -e ./pipeline-adapters/mlrun-pipelines-kfp-common &&\
    pip install -e ./pipeline-adapters/mlrun-pipelines-kfp-v1-8

ENV NO_COLOR=1
