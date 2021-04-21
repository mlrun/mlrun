ARG MLRUN_BASE_IMAGE=mlrun/mlrun:unstable-core

FROM ${MLRUN_BASE_IMAGE}

ENV PIP_NO_CACHE_DIR=1

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -y \
    curl \
    wget \
    build-essential \
    cmake \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt

RUN python -m pip install -r base-image-requirements.txt

ARG MLRUN_MLUTILS_GITHUB_TAG=development
ARG MLRUN_MLUTILS_CACHE_DATE=initial
RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}
