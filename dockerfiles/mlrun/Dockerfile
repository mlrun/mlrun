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

ARG MLRUN_PIP_VERSION=26.0
ARG MLRUN_PYTHON_VERSION=3.11
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest
ARG OMPI_VERSION=4.1.5
ARG MICROMAMBA_VERSION=2.3.0
ARG DOCKER_DEFAULT_PLATFORM=linux/amd64

FROM --platform=${DOCKER_DEFAULT_PLATFORM} mambaorg/micromamba:${MICROMAMBA_VERSION} AS conda-build
ARG MLRUN_PYTHON_VERSION
ARG MLRUN_PIP_VERSION
ARG OMPI_VERSION

ENV MAMBA_DOWNLOAD_THREADS=16
ENV MAMBA_EXTRACT_THREADS=8

RUN micromamba install -y -n base -c conda-forge \
        python=$MLRUN_PYTHON_VERSION openmpi-mpicc=$OMPI_VERSION && \
    micromamba clean -afy
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba config set channel_priority strict && \
    micromamba config set download_threads ${MAMBA_DOWNLOAD_THREADS} && \
    micromamba config set extract_threads ${MAMBA_EXTRACT_THREADS} && \
    micromamba install -y -n base -c conda-forge \
        python=${MLRUN_PYTHON_VERSION} \
        pip=${MLRUN_PIP_VERSION} \
        openmpi-mpicc=${OMPI_VERSION} && \
    micromamba clean -afy

FROM ${MLRUN_UV_IMAGE} AS uv-image
FROM --platform=${DOCKER_DEFAULT_PLATFORM} mlrun_common_image:${MLRUN_PYTHON_VERSION}

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ARG MLRUN_PYTHON_VERSION
ARG MLRUN_PIP_VERSION

COPY --from=conda-build /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"

RUN --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/pip \
    /opt/conda/bin/python -m pip install --upgrade setuptools pip~=${MLRUN_PIP_VERSION}

WORKDIR /mlrun
# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

ARG MLRUN_PYTHON_VERSION
ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Install MLRun:
# no-deps to ignore existing dependencies, just add the locked requirements
# mainly because of how the image is built with conda.
# see https://iguazio.atlassian.net/browse/ML-8712 for example
# where 'tqdm' is given by conda env but will get remove if we omit the `--no-deps`.
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun/locked-requirements.txt,target=locked-requirements.txt \
    uv pip install --no-deps --require-hashes -r locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}

WORKDIR /tmp/mlrun

COPY *.txt *.md *.py *.toml ./
COPY mlrun mlrun
COPY dockerfiles/mlrun-api dockerfiles/mlrun-api

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,id=pip-${MLRUN_PYTHON_VERSION},target=/root/.cache/uv \
    uv pip install .[complete] --python-version ${MLRUN_PYTHON_VERSION}

# Clean up the image:
RUN rm -rf \
    /opt/conda/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
    /usr/local/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
    /var/lib/apt/lists/* \
    /tmp/mlrun

WORKDIR /mlrun
