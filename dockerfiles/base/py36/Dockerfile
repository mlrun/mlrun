ARG MLRUN_PYTHON_VERSION=3.6

FROM python:${MLRUN_PYTHON_VERSION}-slim-buster

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    unzip \
    ca-certificates \
    && update-ca-certificates --fresh \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

ARG OMPI=4.0.3
ENV OPMI=${OMPI}

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY ./dockerfiles/base/requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt

# Install Open MPI
RUN curl -o ./openmpi.zip \
    https://raw.githubusercontent.com/yjb-ds/mlrun-builds/master/archives/openmpi-${OPMI}.zip && \
    unzip openmpi.zip -d /usr && \
    rm -rf ./openmpi.zip

ENV PATH "$PATH:/usr/openmpi/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/openmpi/lib/"

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