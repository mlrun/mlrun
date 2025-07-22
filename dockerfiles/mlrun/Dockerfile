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
ARG MLRUN_UV_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${MLRUN_UV_IMAGE} AS uv-image

FROM gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim

# Update OS packages:
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gcc \
        wget \
        git \
        unzip

WORKDIR /mlrun

# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

# Install MiniConda:
ARG MLRUN_ANACONDA_PYTHON_DISTRIBUTION="-py311"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3${MLRUN_ANACONDA_PYTHON_DISTRIBUTION}_25.1.1-2-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    /opt/conda/bin/conda update --all --use-local --yes && \
    /opt/conda/bin/conda clean --all --quiet --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# Accept terms of service for the conda channel
RUN conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main \
    --channel https://repo.anaconda.com/pkgs/r

# Install Open-MPI:
ARG OMPI_VERSION=4.1.5
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    conda install -y -c conda-forge "openmpi-mpicc=${OMPI_VERSION}"

RUN conda clean -ay

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# need to be redeclared since used in the from
ARG MLRUN_PYTHON_VERSION=3.11
ARG MLRUN_PIP_VERSION=25.0

# Install Python
# use /usr/local/bin/python to ensure system pip is also upgraded
RUN --mount=type=cache,target=/root/.cache/pip \
    /usr/local/bin/python -m pip install --upgrade setuptools pip~=${MLRUN_PIP_VERSION} && \
    conda config --add channels conda-forge && \
    conda install -y python=${MLRUN_PYTHON_VERSION} pip~=${MLRUN_PIP_VERSION} && \
    conda clean -aqy

# Install MLRun:
# no-deps to ignore existing dependencies, just add the locked requirements
# mainly because of how the image is built with conda.
# see https://iguazio.atlassian.net/browse/ML-8712 for example
# where 'tqdm' is given by conda env but will get remove if we omit the `--no-deps`.
# TODO: move tqdm to image requirements
ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun/locked-requirements.txt,target=locked-requirements.txt \
    uv pip install --no-deps --require-hashes -r locked-requirements.txt --python-version ${MLRUN_PYTHON_VERSION}

WORKDIR /tmp/mlrun
COPY *requirements.txt .
COPY setup.py .
COPY packages.py .
COPY README.md .
COPY dependencies.py .
COPY pyproject.toml .
COPY dockerfiles/mlrun-api ./dockerfiles/mlrun-api
COPY mlrun mlrun

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[complete] --python-version ${MLRUN_PYTHON_VERSION}  && rm -rf /tmp/mlrun


# Remove server related files that are not needed
# Remove ensurepip folders from both conda and system python
RUN rm -rf /opt/conda/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
    /usr/local/lib/python${MLRUN_PYTHON_VERSION}/ensurepip \
