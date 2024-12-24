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

RUN apt-get update && apt-get install --no-install-recommends -y \
  gcc \
  git-core \
  graphviz \
  make \
 && rm -rf /var/lib/apt/lists/*

ARG MLRUN_PIP_VERSION=24.2
RUN python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}

WORKDIR /tmp/mlrun

ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/test-system/locked-requirements.txt,target=locked-requirements.txt \
    uv pip sync --require-hashes locked-requirements.txt

COPY . .

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[complete]

RUN mv tests /tests && mv Makefile /Makefile

COPY . /tmp/mlrun

# Clean up all code used for building to simulate user env
RUN rm -rf /tmp/mlrun

CMD ["make",  "test-system"]
