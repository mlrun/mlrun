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

ARG MLRUN_GPU_BASE_IMAGE=quay.io/mlrun/prebaked-cuda:11.7.1-cudnn8-devel-ubuntu20.04

FROM ${MLRUN_GPU_BASE_IMAGE}

# Update OS packages:
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    rm -rf /var/lib/apt/lists/*

# Update base pip packages (usually pip, setuptools, wheel):
# NOTE: here we use /usr/local/bin/python directly to avoid usaing the conda env
# as we want to upgrade the OS python packages
RUN /usr/local/bin/python -m pip  list --disable-pip-version-check --format freeze \
    | grep -v '^\-e' \
    | cut -d'=' -f1 \
    | xargs /usr/local/bin/python -m pip install --upgrade

WORKDIR /mlrun

# non-recursive chmod for the run to be able to create the handler file with any security context
RUN chmod 777 /mlrun

COPY ./requirements.txt ./
COPY ./extras-requirements.txt ./extras-requirements.txt
COPY ./dockerfiles/mlrun/requirements.txt ./mlrun-image-requirements.txt

RUN python -m pip install \
    -r requirements.txt \
    -r extras-requirements.txt \
    -r mlrun-image-requirements.txt

COPY ./dockerfiles/base/requirements.txt ./base-image-requirements.txt
RUN python -m pip install \
    -r base-image-requirements.txt

COPY . .

RUN python -m pip install .[complete]
