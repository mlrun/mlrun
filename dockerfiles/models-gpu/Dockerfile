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
ARG CUDA_VER=11.0

FROM quay.io/mlrun/cuda:${CUDA_VER}-cudnn8-devel-ubuntu18.04

# need to be redeclared since used in the from
ARG CUDA_VER

ENV PIP_NO_CACHE_DIR=1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && \
    apt-mark hold libcublas-dev libcublas10 && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
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

ARG MLRUN_PIP_VERSION=22.0.0
RUN conda config --add channels conda-forge && \
    conda update -n base -c defaults conda && \
    conda install -n base \
        python=3.7 \
        pip~=${MLRUN_PIP_VERSION} \
    && conda clean -aqy

RUN conda install -n base cmake cython \
    && conda clean -aqy

RUN conda install -n base -c rapidsai -c nvidia -c conda-forge rapids=21.06 python=3.7 cudatoolkit=${CUDA_VER} \
    && conda clean -aqy

RUN conda install -n base pytorch::pytorch==1.7.0 pytorch::torchvision==0.8.0 \
    && conda clean -aqy

ARG TENSORFLOW_VERSION=2.4.1
RUN python -m pip install tensorflow~=${TENSORFLOW_VERSION}

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

ARG HOROVOD_VERSION=0.22.1
# TODO: MAKEFLAGS="-j1" work around some transient concurrency problem with installing horovod remove it when
    #   possible (should be safe to remove if it works ~5 times without it)
RUN ldconfig /usr/local/cuda-11.0/targets/x86_64-linux/lib/stubs && \
    MAKEFLAGS="-j1" HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
        python -m pip install horovod~=${HOROVOD_VERSION} && \
    ldconfig

WORKDIR /mlrun

COPY ./requirements.txt ./
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt

RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt

# if we give all the requirements together pip resolver go nuts so separating between the package and the image
# requirements
COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt
RUN python -m pip install \
    -r base-image-requirements.txt

COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt
RUN python -m pip install \
    -r models-image-requirements.txt

COPY ./dockerfiles/models-gpu/requirements.txt ./models-gpu-image-requirements.txt
RUN python -m pip install \
    -r models-gpu-image-requirements.txt

COPY py37 .

# non-recursive chmod for the run to be able to create the handler file with any security context
# at the end of the Dockerfile for docker caching
RUN chmod 777 /mlrun

RUN python -m pip install .[complete]
