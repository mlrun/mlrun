# syntax = docker/dockerfile:1.0-experimental
ARG CUDA_VER=10.1

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG OMPI=4.0.3
ARG MLRUN_GITHUB_REPO=mlrun

ARG MLRUN_PACKAGE_TAG=development
LABEL MLRUN_PACKAGE_TAG=$MLRUN_PACKAGE_TAG

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    bzip2 \
    curl \
    git \
    ibverbs-providers \
    libc6 \
    libglib2.0-0 \
    libibverbs1\
    libjpeg-dev \
    libnvinfer-plugin6=6.0.1-1+cuda10.1 \
    libnvinfer6=6.0.1-1+cuda10.1 \
    libpng-dev \
    librdmacm1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    openssh-client \
    openssh-server \
    unzip \
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
ARG OMPI=4.0.3
ENV OPMI=${OMPI}
RUN curl -o ./openmpi.zip \
    https://raw.githubusercontent.com/yjb-ds/mlrun-builds/master/archives/openmpi-4.0.3.zip && \
    unzip openmpi.zip -d /usr && \
    rm -rf ./openmpi.zip

ENV PATH "$PATH:/usr/openmpi/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/openmpi/lib/"

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
    fsspec \
    lifelines \
    matplotlib \
    numba \
    numpy \
    pandas \
    pyarrow \
    pytest \
    scikit-build \
    scikit-learn \
    scikit-optimize \
    scikit-plot \
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

RUN conda clean -aqy

# do the dask package pinning with pip / faster
COPY ./dockerfiles/models-gpu/requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt

ARG MLRUN_GITHUB_REPO=mlrun
ARG MLRUN_PACKAGE_TAG=development
LABEL MLRUN_TAG=$MLRUN_PACKAGE_TAG
ENV MLRUN_TAG=${MLRUN_PACKAGE_TAG}
RUN python -m pip install --no-cache-dir \
    git+https://github.com/${MLRUN_GITHUB_REPO}/mlrun.git@${MLRUN_PACKAGE_TAG}

ARG MLUTILS_GITHUB_REPO=mlrun
ARG MLRUN_MLUTILS_GITHUB_TAG=development
RUN python -m pip install --no-cache-dir \
    git+https://github.com/${MLUTILS_GITHUB_REPO}/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}