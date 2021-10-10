ARG MLRUN_PYTHON_VERSION=3.7.12

FROM quay.io/mlrun/python:${MLRUN_PYTHON_VERSION}-slim

# need to be redeclared since used in the from
ARG MLRUN_PYTHON_VERSION
ENV PIP_NO_CACHE_DIR=1

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -y \
    build-essential \
    cmake \
    curl \
    gcc \
    git-core \
    graphviz \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlrun

# install miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

RUN conda install python=${MLRUN_PYTHON_VERSION} pip~=20.2.0 && conda clean -aqy

COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./requirements.txt ./
RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt \
    -r base-image-requirements.txt

ARG MLRUN_MLUTILS_GITHUB_TAG=development
ARG MLRUN_MLUTILS_CACHE_DATE=initial
RUN python -m pip install \
    git+https://github.com/mlrun/mlutils.git@${MLRUN_MLUTILS_GITHUB_TAG}
