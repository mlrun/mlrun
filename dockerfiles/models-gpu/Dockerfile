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
ARG MLRUN_BASE_IMAGE=quay.io/mlrun/prebaked-cuda:11.7.0-base-ubuntu20.04

FROM ${MLRUN_BASE_IMAGE}

# install opencv dependencies - putting it here to avoid rebuilding prebaked images
RUN apt-get install ffmpeg libsm6 libxext6  -y

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
COPY ./dockerfiles/models/requirements.txt ./models-image-requirements.txt
COPY ./dockerfiles/models-gpu/requirements.txt ./models-gpu-image-requirements.txt
RUN python -m pip install \
    -r base-image-requirements.txt \
    -r models-image-requirements.txt \
    -r models-gpu-image-requirements.txt

COPY . .

RUN python -m pip install .[complete]
