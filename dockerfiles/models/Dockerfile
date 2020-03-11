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
ARG MLRUN_TAG

FROM ${REPO}/${PREFIX}-base:${MLRUN_TAG}

MAINTAINER yashab@iguazio.com

ARG TF_VER=2.1.0
ARG XGB_VER=1.0.1
ARG LGBM_VER=2.3.1

RUN conda install -n base intelpython3_full python=3.7

RUN conda install -n base -c intel daal4py # tensorflow==${TF_VER}

RUN conda install -n base -c conda-forge seaborn

RUN python -m pip install -U yellowbrick

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
    python -m pip install -e .

RUN cd /repos && \
    git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout v${LGBM_VER} && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../python-package && \
    python -m pip install -e .

ENV USE_DAAL4PY_SKLEARN YES

# RUN conda clean -tipsy \
#     && find /opt/conda/ -type f,l -name '*.a' -delete \
#     && find /opt/conda/ -type f,l -name '*.pyc' -delete \
#     && find /opt/conda/ -type f,l -name '*.js.map' -delete \
#     && rm -rf /opt/conda/pkgs
