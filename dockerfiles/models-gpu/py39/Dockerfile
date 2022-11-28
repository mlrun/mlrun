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
ARG CUDA_VER=11.4.3

#FROM quay.io/mlrun/cuda:${CUDA_VER}-cudnn8-devel-ubuntu18.04
#FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

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

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ARG MLRUN_PIP_VERSION=22.3.0
ARG MLRUN_PYTHON_VERSION=3.9.13

ENV CONDA_OVERRIDE_CUDA ${CUDA_VER}

RUN conda config --add channels conda-forge && \
    conda update -n base -c defaults conda && \
    conda install -n base -c conda-forge \
        python=${MLRUN_PYTHON_VERSION} \
        pip~=${MLRUN_PIP_VERSION} \
        mamba \
    && conda clean -aqy

RUN mamba install -n base -c rapidsai -c nvidia -c pytorch -c conda-forge \
        cmake  \
        cython \
        rapids=22.10 \
        pytorch=1.13.0 \
        torchvision=0.14.0 \
        cxx-compiler \
        nvcc_linux-64 \
        cudnn \
        nccl \
        mpi4py \
        tensorflow=2.8 \
        && conda clean -aqy

ARG OMPI=4.1.0

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -o openmpi-${OMPI}.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI}.tar.gz && \
    tar zxf openmpi-${OMPI}.tar.gz && \
    cd openmpi-${OMPI} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j`nproc` all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN mamba install https://anaconda.org/deepmodeling/horovod/0.25.0/download/linux-64/horovod-0.25.0-py39hfa38a42_0.tar.bz2
RUN horovodrun --check-build

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
