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

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG OMPI=4.0.3
ARG REPO=mlrun

ARG MLRUN_TAG=development
ENV MLRUN_TAG=${MLRUN_TAG}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        cmake \
        curl \
        git \
        ibverbs-providers \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libc6 \
        libglib2.0-0 \
        libibverbs1\
        libjpeg-dev \
        libnvinfer-plugin6=6.0.1-1+cuda10.1 \
        libnvinfer6=6.0.1-1+cuda10.1 \
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
        subversion \
        wget && \
    apt install -y ca-certificates  && \
    update-ca-certificates --fresh && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

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

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda config --add channels pytorch && \
    conda config --add channels conda-forge && \
    conda config --add channels anaconda && \
    conda update -n base conda && \
    conda install -n base \
        bokeh \
        cmake \
        cudatoolkit=10.1 \
        cython \
        dask-kubernetes \
        dask-ml \
        dask-xgboost \
        dask[complete] \
        distributed \
        fsspec \
        matplotlib \
        numba \
        numpy \
        pandas \
        pyarrow \
        pytest \
        pytorch \
        torchvision \
        scikit-build \
        scikit-learn \
        scikit-optimize \
        scipy \
        seaborn \
        tensorflow \
        wrapt

RUN conda install -n base -c rapidsai -c nvidia \
    -c anaconda -c conda-forge -c defaults rapids=0.12

RUN conda clean -afqy

RUN python -m pip install --no-cache-dir \
        face_recognition \
        git+https://github.com/scikit-multiflow/scikit-multiflow \
        gnureadline \
        imutils \
        opencv-contrib-python \
        pyod \
        urllib3==1.24.3 \
        yellowbrick

# xgboost
RUN mkdir /repos && \
    cd repos && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
#    git fetch --all --tags --prune && \
#    git checkout tags/v1.0.2 && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install --no-cache-dir -e .

# lightgbm
# RUN cd /repos && \
#     git clone --recursive https://github.com/microsoft/LightGBM && \
#     cd LightGBM && \
#     git checkout v2.3.1 && \
#     mkdir build && cd build && \
#     cmake .. -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda-10.1/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.1/include/ && \
#     make -j$(nproc) && \
#     make install && \
#     cd ../python-package && \
#     python -m pip install --no-cache-dir -e .

RUN conda clean -aqy

RUN python -c "from matplotlib.font_manager import _rebuild; _rebuild();"

RUN python -m pip install git+https://github.com/${REPO}/mlrun.git@${MLRUN_TAG}    