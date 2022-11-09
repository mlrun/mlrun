# Copyright 2018 Iguazio
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
ARG MLRUN_PYTHON_VERSION=3.9.13

FROM quay.io/mlrun/python:${MLRUN_PYTHON_VERSION}-slim

ENV PIP_NO_CACHE_DIR=1

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

RUN apt-get update && apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        curl \
        g++ \
        git-core \
        gnupg2 \
        graphviz \
        make \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -

RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"

RUN apt update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

ARG MLRUN_PIP_VERSION=22.3.0
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

RUN pip install -e .[complete]
