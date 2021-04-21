
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
ARG MLRUN_PYTHON_VERSION=3.7

FROM python:${MLRUN_PYTHON_VERSION}-slim

# need to be redclared since used in the from
ARG MLRUN_PYTHON_VERSION
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y \
  gcc \
  git-core \
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

RUN conda install python=${MLRUN_PYTHON_VERSION} pip~=20.2.0

COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt
COPY ./requirements.txt ./
RUN python -m pip install \
    -r requirements.txt \
    -r mlrun-image-requirements.txt
