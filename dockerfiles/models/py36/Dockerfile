FROM continuumio/anaconda3:2020.02

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG OMPI=4.0.3

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -yq \
    build-essential \
    cmake \
    curl \
    graphviz \
    openssh-client \
    openssh-server \
    unzip \
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
ARG OMPI=4.0.3
ENV OPMI=${OMPI}
RUN curl -o ./openmpi.zip \
    https://raw.githubusercontent.com/yjb-ds/mlrun-builds/master/archives/openmpi-${OMPI}.zip && \
    unzip openmpi.zip -d /usr && \
    rm -rf ./openmpi.zip

ENV PATH "$PATH:/usr/openmpi/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/openmpi/lib/"

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN conda install -y -c plotly plotly-orca python=3.6

# pytorch caused cuda download, this guarantees cpu version:
RUN python -m pip install --no-cache-dir torch==1.4.0+cpu torchvision==0.5.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# plain vanilla tensorflow isn't compiled optimaly for the architectures
# we are likely to encounter.  this should become an option:
#RUN python -m pip install -U intel-tensorflow==1.14 keras pillow requests

RUN conda install -n base -c anaconda tensorflow=1.14 keras

RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python -m pip install --no-cache-dir horovod

COPY ./dockerfiles/models/requirements.txt ./

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