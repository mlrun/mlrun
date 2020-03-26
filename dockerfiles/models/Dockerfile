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
ARG REPO
ARG PREFIX
ARG MLRUN_TAG=development

FROM continuumio/anaconda3:2020.02

MAINTAINER yashab@iguazio.com

ARG TF_VER=2.1.0
ARG XGB_VER=1.0.1
ARG LGBM_VER=2.3.1

RUN apt update -qqq \
    && apt -y upgrade \
    && apt install -yq --no-install-recommends build-essential \
    graphviz \
    cmake \
    curl \
    libreadline7 \
    libreadline-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    && apt install -y ca-certificates \
    && update-ca-certificates --fresh \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_DIR /etc/ssl/certs

RUN conda update conda

RUN python -m pip install -U --no-cache-dir \
    yellowbrick \
    wrapt \
    kfp \
    pytest \
    pyarrow \
    gnureadline

# xgboost
RUN mkdir /repos && \
    cd repos && \
    git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    git checkout v${XGB_VER} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install --no-cache-dir -e .

RUN cd /repos && \
    git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout v${LGBM_VER} && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install --no-cache-dir -e .

RUN echo ${MLRUN_TAG}

RUN python -m pip install  --ignore-installed --no-cache-dir git+https://github.com/mlrun/mlrun.git@development

#RUN python -m pip uninstall -y pandas && \
#    conda uninstall -n base pandas && \
RUN    conda update -n base -c anaconda pandas

RUN git clone https://github.com/yjb-ds/mlutils.git && \
    cd mlutils && \
    python setup.py develop && \
    python gen_all_configs.py

ENV PYTHONPATH "${PYTHONPATH}:/mlutils"

RUN conda clean -tipsy \
    && find /opt/conda/ -type f,l -name '*.a' -delete \
    && find /opt/conda/ -type f,l -name '*.pyc' -delete \
    && find /opt/conda/ -type f,l -name '*.js.map' -delete \
    && rm -rf /opt/conda/pkgs
