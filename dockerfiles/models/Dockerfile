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

ARG OMPI_VERSION=4.1.4

# Install Open MPI
RUN conda install -c conda-forge openmpi-mpicc=${OMPI_VERSION} && \
    conda clean -aqy

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# pytorch caused cuda download, this guarantees cpu version:
RUN python -m pip install torch==1.13.0+cpu torchvision==0.14.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ARG TENSORFLOW_VERSION=2.9.0

# intel-tensorflow is a slimmer, CPU optimizied distribution of tensorflow by intel
RUN python -m pip install -U intel-tensorflow~=${TENSORFLOW_VERSION} mxnet

ARG HOROVOD_VERSION=0.25.0
RUN HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1  \
    python -m pip install horovod~=${HOROVOD_VERSION} && \
    horovodrun --check-build

# When running mpijob the launcher trigger the worker with something like PATH=/usr/local/bin:$PATH so we need to make
# sure the python in /usr/local/bin is pointing to the conda env python (which "knows" horovod)
# TODO: find nicer solution
RUN ln -sf /opt/conda/bin/python3 /usr/local/bin/python3

COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt

RUN python -m pip install -r models-image-requirements.txt
