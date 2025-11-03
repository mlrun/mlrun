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

ARG MLRUN_PYTHON_VERSION=3.11
ARG UID=1000
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG DOCKER_DEFAULT_PLATFORM=linux/amd64

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM --platform=${DOCKER_DEFAULT_PLATFORM} mlrun_common_image:${MLRUN_PYTHON_VERSION}

ARG MLRUN_PYTHON_VERSION
ARG UID

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

RUN useradd -u $UID mlrun

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends -o Acquire::Check-Valid-Until=false \
        libpq-dev \
        mariadb-client \
        sqlite3 \
        tini

ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun-api/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync  --require-hashes locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}

WORKDIR /tmp/mlrun
COPY *.txt *.md *.py *.toml ./
COPY mlrun mlrun
COPY dockerfiles/mlrun-api dockerfiles/mlrun-api
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install .[complete-api] --python-version ${MLRUN_PYTHON_VERSION}

COPY pipeline-adapters pipeline-adapters
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install ./pipeline-adapters/mlrun-pipelines-kfp-common ./pipeline-adapters/mlrun-pipelines-kfp-v1-8 --python-version ${MLRUN_PYTHON_VERSION}
WORKDIR /mlrun
COPY dockerfiles/mlrun-api/start_api.sh /mlrun/start_api.sh
COPY server/py ./server/py
RUN chmod -R a+rX /mlrun/server/py/schemas/proto

# TODO: remove this once iguazio package is released to PyPI and move to requirements.txt
# Install iguazio package from Test PyPI
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple iguazio~=0.0.1a14 --python-version ${MLRUN_PYTHON_VERSION}

# Remove server related files that are not needed in the base image
# Remove ensurepip folders from both conda and system python
# Remove pip is not needed anymore
RUN python -m pip uninstall pip -y \
    && rm -rf /usr/local/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
    /tmp/mlrun /var/lib/apt/lists/*


VOLUME /mlrun/db
ENV PYTHONPATH=/mlrun/server/py:/mlrun/server/py/schemas/proto
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
ENV MLRUN_SERVICES__SERVICE_NAME=api

# use tini as entrypoint to allow signal handling
# and avoid zombie processes
ENTRYPOINT ["tini", "--"]
CMD ["bash", "/mlrun/start_api.sh"]
