# Copyright 2018 Iguazio
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
#
ARG MLRUN_BASE_IMAGE=mlrun/ml-base:unstable-core

FROM ${MLRUN_BASE_IMAGE}

ARG OMPI=4.1.0

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    curl -o openmpi-${OMPI}.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI}.tar.gz && \
    tar zxf openmpi-${OMPI}.tar.gz && \
    cd openmpi-${OMPI} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j`nproc` all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# pytorch caused cuda download, this guarantees cpu version:
RUN python -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ARG TENSORFLOW_VERSION=2.7.0

# intel-tensorflow is a slimmer, CPU optimizied distribution of tensorflow by intel
RUN python -m pip install -U intel-tensorflow~=${TENSORFLOW_VERSION} mxnet

ARG HOROVOD_VERSION=0.26.1
RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    python -m pip install horovod~=${HOROVOD_VERSION}

# When running mpijob the launcher trigger the worker with something like PATH=/usr/local/bin:$PATH so we need to make
# sure the python in /usr/local/bin is pointing to the conda env python (which "knows" horovod)
# TODO: find nicer solution
RUN ln -sf /opt/conda/bin/python3 /usr/local/bin/python3

COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt

# install rust since it's a dependency of tokenizers
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env

RUN python -m pip install -r models-image-requirements.txt
