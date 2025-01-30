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

# Update OS packages:
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  cmake \
  gcc \
  wget \
  git-core \
  unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /mlrun

# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

# Install MiniConda:
ARG MLRUN_ANACONDA_PYTHON_DISTRIBUTION="-py39"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3${MLRUN_ANACONDA_PYTHON_DISTRIBUTION}_23.11.0-1-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    /opt/conda/bin/conda update --all --use-local --yes && \
    /opt/conda/bin/conda clean --all --quiet --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# Install Open-MPI:
ARG OMPI_VERSION=4.1.5
RUN conda install -y -c conda-forge openmpi-mpicc=${OMPI_VERSION} && \
    conda clean -aqy
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Install Python:
ARG MLRUN_PIP_VERSION=24.2
RUN conda config --add channels conda-forge && \
    conda install -y python=${MLRUN_PYTHON_VERSION} pip~=${MLRUN_PIP_VERSION} && \
    conda clean -aqy


# Install MLRun:
# no-deps to ignore existing dependencies, just add the locked requirements
# mainly because of how the image is built with conda.
# see https://iguazio.atlassian.net/browse/ML-8712 for example
# where 'tqdm' is given by conda env but will get remove if we omit the `--no-deps`.
# TODO: move tqdm to image requirements
ENV UV_SYSTEM_PYTHON=true UV_LINK_MODE=copy
RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dockerfiles/mlrun/locked-requirements.txt,target=locked-requirements.txt \
    uv pip install --no-deps --require-hashes -r locked-requirements.txt

COPY . .

RUN --mount=from=uv-image,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[complete]
