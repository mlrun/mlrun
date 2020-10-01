FROM continuumio/anaconda3:2020.02

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

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

ARG OMPI=4.0.3

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

RUN conda install -y -c plotly plotly-orca

RUN conda clean -ayq

# pytorch caused cuda download, this guarantees cpu version:
RUN python -m pip install torch==1.4.0+cpu torchvision==0.5.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN python -m pip install -U intel-tensorflow mxnet

RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python -m pip install horovod

WORKDIR /mlrun

COPY ./dockerfiles/models/requirements.txt ./

RUN python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install .

ARG MLRUN_MLUTILS_GITHUB_TAG=development
ARG MLRUN_MLUTILS_CACHE_DATE=initial
RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}
