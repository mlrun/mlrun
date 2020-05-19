ARG MLRUN_PYTHON_VERSION=3.6

FROM python:${MLRUN_PYTHON_VERSION}-slim-buster

LABEL maintainer="yashab@iguazio.com"
LABEL org="iguazio.com"

ARG MLRUN_GITHUB_REPO=mlrun

ARG MLRUN_PACKAGE_TAG=development

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    ca-certificates \
    && update-ca-certificates --fresh \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

COPY ./dockerfiles/base/requirements.txt ./

RUN python -m pip install -r requirements.txt

RUN python -m pip install \
    git+https://github.com/${MLRUN_GITHUB_REPO}/mlrun.git@${MLRUN_PACKAGE_TAG}

RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@05657d9907779de0d823b9f0d7466a17c6490460