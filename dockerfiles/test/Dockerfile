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
ARG MLRUN_PYTHON_VERSION=3.7

FROM python:${MLRUN_PYTHON_VERSION}-slim

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

RUN apt-get update && apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        g++ \
        git-core \
        gnupg2 \
        make \
        software-properties-common

RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -

RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"

RUN apt update && apt-get install -y docker-ce-cli

RUN python -m pip install --upgrade --no-cache pip

WORKDIR /mlrun

COPY ./requirements.txt ./
COPY ./dockerfiles/mlrun-api/requirements.txt ./mlrun-api-requirements.txt
COPY ./dev-requirements.txt ./
COPY ./docs/requirements.txt ./docs-requirements.txt

RUN pip install \
    -r requirements.txt \
    -r mlrun-api-requirements.txt \
    -r dev-requirements.txt \
    -r docs-requirements.txt

COPY . .
