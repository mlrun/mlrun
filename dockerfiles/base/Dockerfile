ARG MLRUN_PYTHON_VERSION=3.7

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
    ca-certificates \
    && update-ca-certificates --fresh \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

WORKDIR /mlrun

COPY ./dockerfiles/base/requirements.txt ./

RUN python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install .

ARG MLRUN_MLUTILS_GITHUB_TAG=development
ARG MLRUN_MLUTILS_CACHE_DATE=initial
RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}
