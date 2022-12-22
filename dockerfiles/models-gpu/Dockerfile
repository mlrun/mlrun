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
ARG CUDA_VER=11.7.0

FROM quay.io/mlrun/cuda:11.7.0-base-ubuntu20.04

# need to be redeclared since used in the from
ARG CUDA_VER

ENV PIP_NO_CACHE_DIR=1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
        gcc \
        cmake \
        curl \
        git-core \
        graphviz \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -f -b -p /opt/conda && \
    rm ~/installconda.sh && \
    conda init bash

ARG MLRUN_PIP_VERSION=22.3.0
ARG MLRUN_PYTHON_VERSION=3.9.13

ENV CONDA_OVERRIDE_CUDA ${CUDA_VER}

RUN conda config --add channels conda-forge && \
    conda update -n base -c defaults conda && \
    conda install -n base \
        python=${MLRUN_PYTHON_VERSION} \
        pip~=${MLRUN_PIP_VERSION} \
    && conda clean -aqy

RUN conda install -n base -c rapidsai -c nvidia -c pytorch -c conda-forge \
        cmake  \
        cudatoolkit=${CUDA_VER} \
        cudnn \
        cxx-compiler=1.5.1 \
        cython \
        make \
        nccl \
        python=${MLRUN_PYTHON_VERSION} \
        pytorch=1.13 \
        rapids=22.10 \
        tensorflow=2.9 \
        torchvision=0.14 \
        openmpi-mpicc=4.1.4 \
    && conda clean -aqy

ARG HOROVOD_VERSION=0.25.0

RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
        python -m pip install horovod~=${HOROVOD_VERSION} && \
    horovodrun --check-build

WORKDIR /mlrun

# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

COPY ./requirements.txt ./
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt

RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt

COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt
COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt
COPY ./dockerfiles/models-gpu/requirements.txt ./models-gpu-image-requirements.txt
RUN python -m pip install \
    -r base-image-requirements.txt \
    -r models-image-requirements.txt \
    -r models-gpu-image-requirements.txt

COPY . .

RUN python -m pip install .[complete]
