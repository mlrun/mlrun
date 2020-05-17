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
ARG CUDA_VER=10.1

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG OMPI=4.0.3
ARG MLRUN_GITHUB_REPO=mlrun

ARG MLRUN_PACKAGE_TAG=development
LABEL MLRUN_TAG=$MLRUN_PACKAGE_TAG
ENV MLRUN_TAG=${MLRUN_PACKAGE_TAG}

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
        scikit-build \
        scikit-learn \
        scikit-optimize \
        scipy \
        seaborn \
        tensorflow-gpu \
        wrapt

RUN conda install -n base -c rapidsai -c nvidia \
    -c anaconda -c conda-forge -c defaults rapids=0.12

RUN conda install -n base -c pytorch pytorch torchvision cudatoolkit=10.1

RUN ldconfig /usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
        python -m pip install --no-cache-dir horovod && \
    ldconfig

RUN python -m pip install --no-cache-dir \
        face_recognition \
        git+https://github.com/scikit-multiflow/scikit-multiflow \
        gnureadline \
        imutils \
        opencv-contrib-python \
        pyod \
        urllib3==1.24.3 \
        yellowbrick

RUN python -c "from matplotlib.font_manager import _rebuild; _rebuild();"

RUN conda clean -aqy

RUN python -m pip install git+https://github.com/${MLRUN_GITHUB_REPO}/mlrun.git@${MLRUN_PACKAGE_TAG}

RUN python -m pip install --ignore-installed --no-cache-dir \
    blosc==1.7.0 \
    cloudpickle==1.1.1 \
    dask==2.12.0 \
    distributed==2.14.0 \
    lz4==2.2.1 \
    msgpack==1.0.0 \
    tornado==6.0.4

RUN python -m pip install urllib3==1.25.8 kubernetes==10.0.1

RUN python -m pip install lifelines scikit-plot

RUN python -m pip install git+https://github.com/${MLRUN_GITHUB_REPO}/mlutils.git@development
