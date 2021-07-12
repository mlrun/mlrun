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
ARG CUDA_VER=11.2.2

FROM nvidia/cuda:${CUDA_VER}-cudnn8-devel-ubuntu18.04

# need to be redeclared since used in the from
ARG CUDA_VER

ENV PIP_NO_CACHE_DIR=1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-mark hold libcublas-dev libcublas10 && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        wget \
        graphviz && \
    rm -rf /var/lib/apt/lists/*

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

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda config --add channels conda-forge && \
    conda config --add channels anaconda && \
    conda update -n base conda && \
    conda install -n base \
        cmake \
        cudatoolkit=${CUDA_VER} \
        cython \
        tensorflow-gpu=2.2.0 \
    && conda clean -aqy

RUN conda install -n base -c rapidsai -c nvidia \
    -c anaconda -c conda-forge -c defaults rapids=0.12 python=3.7 \
    && conda clean -aqy

RUN conda install -n base -c pytorch pytorch torchvision cudatoolkit=10.1 \
    && conda clean -aqy

RUN python -m pip install --upgrade pip~=20.2.0

# TODO: MAKEFLAGS="-j1" work around some transient concurrency problem with installing horovod remove it when
    #   possible (should be safe to remove if it works ~5 times without it)
RUN ldconfig /usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs && \
    MAKEFLAGS="-j1" HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
        python -m pip install 'horovod~=0.20.0' && \
    ldconfig

WORKDIR /mlrun

COPY ./requirements.txt ./
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt
COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt
COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt
COPY ./dockerfiles/models-gpu/requirements.txt ./models-gpu-image-requirements.txt

RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt \
    -r base-image-requirements.txt \
    -r models-image-requirements.txt \
    -r models-gpu-image-requirements.txt

COPY . .
RUN python -m pip install .[complete]

ARG MLRUN_MLUTILS_GITHUB_TAG=development
ARG MLRUN_MLUTILS_CACHE_DATE=initial
RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}
