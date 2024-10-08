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

# Update OS packages:
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

# Update base pip packages (usually pip, setuptools, wheel):
RUN python -m pip  list --disable-pip-version-check --format freeze \
    | grep -v '^\-e' \
    | cut -d'=' -f1 \
    | xargs python -m pip install --upgrade

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  cmake \
  gcc \
  wget \
  git-core \
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

ENV PATH /opt/conda/bin:$PATH

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
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./requirements.txt ./
RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt

COPY . .
RUN python -m pip install .[complete]
