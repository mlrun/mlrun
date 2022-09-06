
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
ARG MLRUN_PYTHON_VERSION=3.7.13

FROM quay.io/mlrun/python:${MLRUN_PYTHON_VERSION}-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y \
  gcc \
  git-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /mlrun

# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

ARG MLRUN_PIP_VERSION=22.0.0
RUN python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}

COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./requirements.txt ./
RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt

COPY . .
RUN python -m pip install .[complete]
