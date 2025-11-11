# Copyright 2025 Iguazio
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

ARG MLRUN_PYTHON_VERSION=3.11
ARG DOCKER_DEFAULT_PLATFORM=linux/amd64

FROM --platform=${DOCKER_DEFAULT_PLATFORM} gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get update -q && \
    apt-get -y upgrade -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libffi-dev \
        python3-dev \
        unzip \
    && python -m pip list --format=freeze \
        | grep -v '^\-e' \
        | cut -d = -f 1 \
        | xargs python -m pip install --upgrade
