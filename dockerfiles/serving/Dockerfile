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
ARG MLRUN_DOCKER_REPO=mlrun
ARG MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX=ml-
ARG MLRUN_DOCKER_TAG

FROM ${MLRUN_DOCKER_REPO}/${MLRUN_ML_DOCKER_IMAGE_NAME_PREFIX}base:${MLRUN_DOCKER_TAG}

RUN python3 -m pip install --no-cache-dir -U kfserving
