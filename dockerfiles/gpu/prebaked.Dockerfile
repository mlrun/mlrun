# Copyright 2023 Iguazio
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

ARG CUDA_VER

# CUDA Based image (including cudnn and cuda-toolkit):
FROM gcr.io/iguazio/nvidia/cuda:$CUDA_VER

# Update apt:
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt update -qqq --fix-missing \
    && apt upgrade -y \
    && apt install -y \
    build-essential \
    cmake \
    curl \
    gcc \
    git-core \
    graphviz \
    wget \
    bzip2 \
    ca-certificates \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install MiniConda (Python 3.9):
ARG MLRUN_ANACONDA_PYTHON_DISTRIBUTION="-py39"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3${MLRUN_ANACONDA_PYTHON_DISTRIBUTION}_23.11.0-1-Linux-x86_64.sh -O ~/installconda.sh && \
    /bin/bash ~/installconda.sh -b -p /opt/conda && \
    rm ~/installconda.sh && \
    /opt/conda/bin/conda update --all --use-local --yes && \
    /opt/conda/bin/conda clean --all --quiet --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# Setup environment variables:
ENV PIP_NO_CACHE_DIR=1
ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Install Open-MPI:
ARG OMPI_VERSION=4.1.5
RUN conda install -c conda-forge openmpi-mpicc=${OMPI_VERSION} && \
    conda clean -aqy
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_opal_cuda_support="true"
