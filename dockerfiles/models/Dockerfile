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
FROM continuumio/anaconda3:2020.02

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG OMPI=4.0.3
ARG REPO=mlrun

ARG MLRUN_TAG=development
LABEL MLRUN_TAG=$MLRUN_TAG
ENV MLRUN_TAG=${MLRUN_TAG}

ARG TF_VER=2.1.0
ARG XGB_VER=1.0.2
ARG LGBM_VER=2.3.1

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -yq \
        build-essential \
        cmake \
        curl \
        graphviz \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libreadline-dev \
        libreadline7 \
        openssh-client \
        openssh-server \
        subversion \
    && apt install -y ca-certificates \
    && update-ca-certificates --fresh \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

RUN mkdir -p /var/run/sshd && \
    # Allow OpenSSH to talk to containers without asking for confirmation
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -o openmpi-${OMPI}.tar.gz https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OMPI}.tar.gz && \
    tar zxf openmpi-${OMPI}.tar.gz && \
    cd openmpi-${OMPI} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j`nproc` all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN conda config --add channels pytorch && \
    conda config --add channels conda-forge && \
    conda config --add channels anaconda && \
    conda install -y \
        bokeh \
        dask-kubernetes \
        dask-ml \
        dask-xgboost \
        dask[complete] \
        distributed \
        fsspec \
        pyarrow \
        pytest \
        scikit-optimize \
        wrapt

# pytorch caused cuda download, this guarantees cpu version:
RUN python -m pip install torch==1.4.0+cpu torchvision==0.5.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html

# plain vanilla tensorflow isn't compiled optimaly for the architectures
# we are likely to encounter.  this should become an option:
RUN python -m pip install -U intel-tensorflow

RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python -m pip install --no-cache-dir horovod

RUN python -m pip install -U --no-cache-dir \
        git+https://github.com/scikit-multiflow/scikit-multiflow.git \
        gnureadline \
        pyod \
        urllib3==1.24.3 \
        yellowbrick 

# xgboost
RUN mkdir /repos && \
    cd repos && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    # git fetch --all --tags --prune && \
    # git checkout tags/v1.0.2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install --no-cache-dir -e .

RUN cd /repos && \
    git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout v${LGBM_VER} && \
    mkdir build && cd build && \
    cmake -DUSE_MPI=1 .. && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install --no-cache-dir -e .

RUN conda clean -ayq

RUN python -c "from matplotlib.font_manager import _rebuild; _rebuild();"

RUN python -m pip install git+https://github.com/${REPO}/mlrun.git@${MLRUN_TAG}

# remove dask verify version mismatch warnings
RUN python -m pip install --no-cache-dir \
        blosc==1.7.0 \
        cloudpickle==1.1.1 \
        dask==2.12.0 \
        distributed==2.14.0 \
        lz4==2.2.1 \
        msgpack==1.0.0 \
        tornado==6.0.4
