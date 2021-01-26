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

LABEL maintainer="yaronh@iguazio.com"
LABEL org="iguazio.com"

RUN apt-get update && apt-get install -y \
  gcc \
  git-core \
  sqlite3 \
  procps \
  vim

RUN python -m pip install --upgrade --no-cache pip

WORKDIR /mlrun

COPY ./dockerfiles/mlrun-api/requirements.txt ./mlrun-api-requirements.txt

COPY ./requirements.txt ./

RUN python -m pip install \
    -r requirements.txt \
    -r mlrun-api-requirements.txt

ENV MLRUN_httpdb__dirpath=/mlrun/db
ENV MLRUN_httpdb__port=8080
COPY . .
VOLUME /mlrun/db

CMD ["python",  "-m",  "mlrun",  "db"]
