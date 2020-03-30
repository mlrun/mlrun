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

ARG MLRUN_TAG=development
ENV MLRUN_TAG=$MLRUN_TAG

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    nano \
    wget \
    bzip2 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    python3 \
    python3-dev \
    python3-distutils \
    libxrender1 \
    mercurial \
    subversion \
    libreadline7 \
    libreadline-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    python3-libnvinfer \
    python3-libnvinfer-dev && \
    apt install -y ca-certificates  && \
    update-ca-certificates --fresh && \
    apt clean && \
    apt autoremove

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
RUN python3 -m pip install --no-cache-dir tensorflow h5py torch torchvision

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.3.tar.gz && \
    tar zxf openmpi-4.0.3.tar.gz && \
    cd openmpi-4.0.3 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j`nproc` all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# # Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python3 -m pip install --no-cache-dir horovod && \
    ldconfig

# # Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# # Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# # Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/horovod/horovod/trunk/examples && \
    rm -rf /examples/.svn

RUN python3 -m pip install --no-cache-dir git+https://github.com/mlrun/mlrun.git@${MLRUN_TAG}

RUN python3 -m pip install --no-cache-dir scikit-build cmake face_recognition opencv-contrib-python imutils

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 0 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
    update-alternatives  --set python /usr/bin/python3.6

WORKDIR "/examples"
