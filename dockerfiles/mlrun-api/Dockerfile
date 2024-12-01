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


ARG UID=1000
# this creates both user and group with the same id
RUN useradd -u $UID mlrun

# Update OS packages:
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

RUN apt-get update && apt-get install --no-install-recommends -y \
  tini \
  gcc \
  git-core \
  procps \
  sqlite3 \
  vim \
  mariadb-client \
  curl \
  ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv
ARG MLRUN_UV_VERSION=0.5.5
ADD https://astral.sh/uv/${MLRUN_UV_VERSION}/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"
ENV UV_SYSTEM_PYTHON=true

WORKDIR /mlrun

COPY ./dockerfiles/mlrun-api/locked-requirements.txt ./locked-requirements.txt

RUN uv pip sync --require-hashes locked-requirements.txt

ENV MLRUN_HTTPDB__DSN='sqlite:////mlrun/db/mlrun.db?check_same_thread=false'
ENV MLRUN_HTTPDB__LOGS_PATH=/mlrun/db/logs
ENV MLRUN_HTTPDB__DIRPATH=/mlrun/db
ENV MLRUN_HTTPDB__PORT=8080
ENV MLRUN_HTTPDB__HTTP_CONNECTION_TIMEOUT_KEEP_ALIVE=11
ENV MLRUN_IS_API_SERVER=true
ENV MLRUN_IGNORE_ENV_FILE=true

# env vars for uvicorn (https://www.uvicorn.org/settings)
ENV UVICORN_PORT=${MLRUN_HTTPDB__PORT}
ENV UVICORN_TIMEOUT_KEEP_ALIVE=${MLRUN_HTTPDB__HTTP_CONNECTION_TIMEOUT_KEEP_ALIVE}

COPY ./pipeline-adapters ./pipeline-adapters

RUN uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-common && \
    uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-v1-8

COPY . .
ENV PYTHONPATH=/mlrun/server/py

RUN uv pip install .[complete-api]

VOLUME /mlrun/db

ENV MLRUN_SERVICES__SERVICE_NAME=api

# use tini as entrypoint to allow signal handling
# and avoid zombie processes
ENTRYPOINT ["tini", "--"]

CMD ["bash", "./dockerfiles/mlrun-api/start_api.sh"]
