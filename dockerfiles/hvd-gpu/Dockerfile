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
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG PYTHON_VER=3.8
ARG OMPI=4.0.3
ARG REPO=mlrun

ARG MLRUN_TAG=development
ENV MLRUN_TAG=${MLRUN_TAG}

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    python${PYTHON_VER} \
    python${PYTHON_VER}-dev \
    python${PYTHON_VER}-distutils \
        build-essential \
        bzip2 \
        cmake \
        curl \
        git \
        ibverbs-providers \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libglib2.0-0 \
        libibverbs1 \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libreadline-dev \
        libreadline7 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        openssh-server \
        python3-libnvinfer \
        python3-libnvinfer-dev \
        subversion \
        wget && \
    apt install -y ca-certificates  && \
    update-ca-certificates --fresh && \
    apt clean && \
    apt autoremove

RUN mkdir -p /var/run/sshd && \
    # Allow OpenSSH to talk to containers without asking for confirmation
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

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

RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 0 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 1 && \
    update-alternatives  --set python /usr/bin/python${PYTHON_VER}

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN python -m pip install --no-cache-dir \
        tensorflow \
        h5py \
        torch \
        torchvision \
        scikit-learn \
        numpy \
        pandas \
        scikit-build \
        cmake \
        face_recognition \
        opencv-contrib-python \
        imutils

# # Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
        python -m pip install --no-cache-dir horovod && \
    ldconfig

RUN python -m pip install --no-cache-dir git+https://github.com/${REPO}/mlrun.git@${MLRUN_TAG}

# WORKDIR "/examples"