# Copyright 2020 Iguazio
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
ARG PYTHON_VER=3.8

FROM python:${PYTHON_VER}-slim-buster

ENV PYTHON_VER=$PYTHON_VER

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG REPO=mlrun

ARG MLRUN_TAG=development
ENV MLRUN_TAG=${MLRUN_TAG}

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    ca-certificates \
    && update-ca-certificates --fresh \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

RUN python -m pip install --ignore-installed --no-cache-dir git+https://github.com/${REPO}/mlrun.git@${MLRUN_TAG}
