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

ARG MLRUN_GPU_BASE_IMAGE=quay.io/mlrun/prebaked-cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${MLRUN_UV_IMAGE} AS uv-image

FROM ${MLRUN_GPU_BASE_IMAGE}

# Update OS packages:
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

# Update existing requirements to remedy basic security vulnerabilities
# e.g.: wheel, requests, etc.
RUN conda update --all --use-local --yes && conda clean --all --quiet --yes

WORKDIR /mlrun

# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy

# do not require hashes as PyHive is installed via remote repo
# and do not have a hash set in the locked-requirements.txt
# therefor, verify the hashes but do not require them.
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/gpu/locked-requirements.txt,target=locked-requirements.txt \
    uv pip install --verify-hashes -r locked-requirements.txt

COPY . .

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[complete]
