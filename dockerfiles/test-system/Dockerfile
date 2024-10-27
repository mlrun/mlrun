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
ARG MLRUN_PYTHON_VERSION=3.9

FROM gcr.io/iguazio/python:${MLRUN_PYTHON_VERSION}-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y \
  gcc \
  git-core \
  graphviz \
  make \
 && rm -rf /var/lib/apt/lists/*

ARG MLRUN_PIP_VERSION=24.2
RUN python -m pip install --upgrade pip~=${MLRUN_PIP_VERSION}

COPY ./dockerfiles/mlrun-api/requirements.txt /tmp/mlrun-requirements/mlrun-api-requirements.txt
COPY ./extras-requirements.txt /tmp/mlrun-requirements
COPY ./requirements.txt /tmp/mlrun-requirements
COPY ./dev-requirements.txt /tmp/mlrun-requirements
RUN python -m pip install \
  -r /tmp/mlrun-requirements/requirements.txt \
  -r /tmp/mlrun-requirements/dev-requirements.txt \
  -r /tmp/mlrun-requirements/extras-requirements.txt \
  -r /tmp/mlrun-requirements/mlrun-api-requirements.txt
COPY . /tmp/mlrun
RUN cd /tmp/mlrun && python -m pip install ".[complete]" && mv tests /tests && mv Makefile /Makefile

# Clean up all code used for building to simulate user env
RUN rm -rf /tmp/mlrun-requirements && rm -rf /tmp/mlrun

CMD ["make",  "test-system"]
