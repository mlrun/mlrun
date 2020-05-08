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
ARG PYTHON_VER=3.6

FROM python:${PYTHON_VER}-slim

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

ENV PYTHON_VER=${PYTHON_VER}

RUN apt-get update && apt-get install -y \
  gcc \
  git-core \
  vim

RUN python -m pip install --upgrade --no-cache pip

WORKDIR /mlrun

COPY ./dockerfiles/*requirements.txt ./

COPY ./requirements.txt ./

RUN python -m pip install \
    -r dask-requirements.txt \
    -r api-requirements.txt \
    -r requirements.txt

ENV MLRUN_httpdb__dirpath=/mlrun/db
ENV MLRUN_httpdb__port=8080
COPY . .
VOLUME /mlrun/db

CMD ["/mlrun/scripts/api/docker-start.sh"]
